"""Point-cloud dataset model bundling points with space metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lsfm_cell_mapping.pointcloud.centroids import POINTCLOUD_REQUIRED_COLUMNS
from lsfm_cell_mapping.pointcloud.metadata import PointCloudSpace


@dataclass
class PointCloudDataset:
    """A point cloud together with the space metadata needed to interpret it."""

    subject_name: str
    points: pd.DataFrame
    space: PointCloudSpace

    @classmethod
    def from_files(
        cls,
        csv_path: Path,
        json_path: Path,
        *,
        subject_name: str | None = None,
    ) -> "PointCloudDataset":
        """Load a point-cloud dataset from a CSV and matching space JSON."""

        points = pd.read_csv(csv_path)
        space = PointCloudSpace.from_json(json_path)

        if subject_name is None:
            subject_name = _infer_subject_name_from_pointcloud_path(csv_path)

        return cls(
            subject_name=subject_name,
            points=points,
            space=space,
        )

    def validate(self) -> None:
        """Validate the dataset schema and coordinate bounds."""

        _validate_required_columns(self.points)
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


def _infer_subject_name_from_pointcloud_path(csv_path: Path) -> str:
    """Infer a subject name from a standard point-cloud CSV filename."""

    suffix = "_pointcloud"
    stem = csv_path.stem
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def _validate_required_columns(points: pd.DataFrame) -> None:
    """Check that the point table contains the canonical required columns."""

    missing = [column for column in POINTCLOUD_REQUIRED_COLUMNS if column not in points.columns]
    if missing:
        raise ValueError(f"Point cloud is missing required columns: {missing}")


def _validate_axis_metadata(space: PointCloudSpace) -> None:
    """Check that axis metadata lengths are internally consistent."""

    if len(space.axis_labels) != 3:
        raise ValueError(f"axis_labels must have length 3, got {space.axis_labels}")
    if len(space.shape) != 3:
        raise ValueError(f"shape must have length 3, got {space.shape}")
    if len(space.resolution_um) != 3:
        raise ValueError(
            f"resolution_um must have length 3, got {space.resolution_um}"
        )


def _validate_indexing(space: PointCloudSpace) -> None:
    """Check that indexing is one of the supported conventions."""

    valid_indexing = {"one_based", "zero_based"}
    if space.indexing not in valid_indexing:
        raise ValueError(
            f"indexing must be one of {sorted(valid_indexing)}, got {space.indexing}"
        )


def _validate_point_bounds(points: pd.DataFrame, space: PointCloudSpace) -> None:
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
