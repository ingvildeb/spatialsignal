"""Shared dataset model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatialsignal.models.metadata import DatasetMetadata, SpaceDefinition


POINTCLOUD_REQUIRED_COLUMNS = ["detection_id", "seg_num", "x", "y", "z"]


@dataclass
class PointCloudDataset:
    """A point cloud together with the metadata needed to interpret it."""

    subject_name: str
    points: pd.DataFrame
    metadata: DatasetMetadata

    @classmethod
    def from_files(
        cls,
        table_path: Path,
        json_path: Path,
        *,
        subject_name: str | None = None,
    ) -> "PointCloudDataset":
        """Load a point-cloud dataset from Parquet and matching metadata JSON."""

        points = pd.read_parquet(table_path)
        metadata = DatasetMetadata.from_json(json_path)

        if subject_name is None:
            subject_name = _infer_subject_name_from_pointcloud_path(table_path)

        return cls(
            subject_name=subject_name,
            points=points,
            metadata=metadata,
        )

    @property
    def space(self) -> SpaceDefinition:
        """Convenience accessor for the dataset spatial definition."""

        return self.metadata.space

    def validate(self) -> None:
        """Validate the dataset schema and coordinate bounds."""

        _validate_required_columns(self.points)
        self.validate_spatial_points()

    def validate_spatial_points(self) -> None:
        """Validate the spatial coordinate contract for any point-based dataset."""

        _validate_spatial_coordinate_columns(self.points, self.space)
        _validate_axis_metadata(self.space)
        _validate_indexing(self.space)
        _validate_point_bounds(self.points, self.space)

    def summary(self) -> dict[str, Any]:
        """Return a compact dataset summary useful for QC and inspection."""

        self.validate()

        summary: dict[str, Any] = {
            "subject_name": self.subject_name,
            "space_name": self.space.space_name,
            "orientation": self.space.orientation,
            "n_points": int(len(self.points)),
            "indexing": self.space.indexing,
        }

        for axis_label in self.space.axis_labels:
            summary[f"{axis_label}_min"] = int(self.points[axis_label].min())
            summary[f"{axis_label}_max"] = int(self.points[axis_label].max())

        return summary


@dataclass
class VoxelMap:
    """A voxelized spatial dataset together with its metadata.

    The canonical in-memory axis order is ``data[x, y, z]`` so that voxel-map
    indexing matches the package's explicit spatial metadata and point-cloud
    coordinate conventions directly.
    """

    subject_name: str
    data: np.ndarray
    metadata: DatasetMetadata

    @classmethod
    def from_files(
        cls,
        array_path: Path,
        json_path: Path,
        *,
        subject_name: str,
    ) -> "VoxelMap":
        """Load a voxel map from its NumPy array and metadata sidecar."""

        data = np.load(array_path, allow_pickle=False)
        metadata = DatasetMetadata.from_json(json_path)
        if tuple(data.shape) != tuple(metadata.space.shape):
            raise ValueError(
                "Voxel-map array shape does not match metadata space shape: "
                f"{data.shape} vs {tuple(metadata.space.shape)}"
            )
        return cls(subject_name=subject_name, data=data, metadata=metadata)

    @classmethod
    def from_nifti(
        cls,
        nifti_path: Path,
        json_path: Path,
        *,
        subject_name: str,
    ) -> "VoxelMap":
        """Load a voxel map from NIfTI data and its metadata sidecar."""

        try:
            import nibabel as nib
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading NIfTI voxel maps requires nibabel"
            ) from exc

        image = nib.load(str(nifti_path))
        data = np.asarray(image.dataobj)
        metadata = DatasetMetadata.from_json(json_path)
        if tuple(data.shape) != tuple(metadata.space.shape):
            raise ValueError(
                "Voxel-map NIfTI shape does not match metadata space shape: "
                f"{data.shape} vs {tuple(metadata.space.shape)}"
            )
        return cls(subject_name=subject_name, data=data, metadata=metadata)

    @property
    def space(self) -> SpaceDefinition:
        """Convenience accessor for the voxel map spatial definition."""

        return self.metadata.space

    def summary(self) -> dict[str, Any]:
        """Return a compact summary useful for QC and inspection."""

        return {
            "subject_name": self.subject_name,
            "space_name": self.space.space_name,
            "orientation": self.space.orientation,
            "shape": tuple(int(v) for v in self.data.shape),
            "dtype": str(self.data.dtype),
            "representation_kind": self.metadata.representation.kind,
            "representation_type": self.metadata.representation.representation_type,
            "nonzero_voxels": int(np.count_nonzero(self.data)),
            "sum": float(self.data.sum()),
            "max": float(self.data.max()) if self.data.size else 0.0,
        }


def _infer_subject_name_from_pointcloud_path(table_path: Path) -> str:
    """Infer a subject name from a standard point-cloud table filename."""

    suffix = "_pointcloud"
    stem = table_path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _validate_required_columns(points: pd.DataFrame) -> None:
    """Check that the point table contains the canonical required columns."""

    missing = [column for column in POINTCLOUD_REQUIRED_COLUMNS if column not in points.columns]
    if missing:
        raise ValueError(f"Point cloud is missing required columns: {missing}")


def _validate_spatial_coordinate_columns(
    points: pd.DataFrame,
    space: SpaceDefinition,
) -> None:
    """Check that the point table contains the spatial coordinate columns for the space."""

    missing = [axis_label for axis_label in space.axis_labels if axis_label not in points.columns]
    if missing:
        raise ValueError(
            f"Point table is missing required spatial coordinate columns: {missing}"
        )


def _validate_axis_metadata(space: SpaceDefinition) -> None:
    """Check that axis metadata lengths are internally consistent."""

    if len(space.axis_labels) != 3:
        raise ValueError(f"axis_labels must have length 3, got {space.axis_labels}")
    if len(space.shape) != 3:
        raise ValueError(f"shape must have length 3, got {space.shape}")
    if len(space.resolution_um) != 3:
        raise ValueError(
            f"resolution_um must have length 3, got {space.resolution_um}"
        )


def _validate_indexing(space: SpaceDefinition) -> None:
    """Check that indexing is one of the supported conventions."""

    valid_indexing = {"one_based", "zero_based"}
    if space.indexing not in valid_indexing:
        raise ValueError(
            f"indexing must be one of {sorted(valid_indexing)}, got {space.indexing}"
        )


def _validate_point_bounds(points: pd.DataFrame, space: SpaceDefinition) -> None:
    """Check that point coordinates fall within the declared space shape."""

    lower_bound = 1 if space.indexing == "one_based" else 0

    for axis_label, axis_size in zip(space.axis_labels, space.shape, strict=True):
        values = points[axis_label]
        if values.empty:
            continue

        if (values < lower_bound).any():
            raise ValueError(
                f"Point cloud contains {axis_label} values below the {space.indexing} lower bound of {lower_bound}"
            )

        upper_bound = axis_size if space.indexing == "one_based" else axis_size - 1
        if (values > upper_bound).any():
            raise ValueError(
                f"Point cloud contains {axis_label} values above the {space.indexing} upper bound of {upper_bound}"
            )
