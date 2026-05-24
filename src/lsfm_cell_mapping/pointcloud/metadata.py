"""Point-cloud space metadata helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import tifffile


@dataclass
class PointCloudSpace:
    """Space-only metadata describing how to interpret a point cloud."""

    schema_name: str
    schema_version: str
    space_name: str
    orientation: str
    axis_labels: list[str]
    indexing: str
    units: str
    shape: list[int]
    resolution_um: list[float]

    def to_dict(self) -> dict[str, Any]:
        """Convert the metadata object to a plain dictionary."""

        return asdict(self)

    def to_json(self, output_path: Path) -> None:
        """Write the metadata object to JSON."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PointCloudSpace":
        """Construct metadata from a dictionary."""

        return cls(
            schema_name=data["schema_name"],
            schema_version=data["schema_version"],
            space_name=data["space_name"],
            orientation=data["orientation"],
            axis_labels=data["axis_labels"],
            indexing=data["indexing"],
            units=data["units"],
            shape=data["shape"],
            resolution_um=data["resolution_um"],
        )

    @classmethod
    def from_json(cls, json_path: Path) -> "PointCloudSpace":
        """Load metadata from a JSON file."""

        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)

    @classmethod
    def from_mask_files(
        cls,
        *,
        space_name: str,
        orientation: str,
        resolution_um: list[float],
        indexing: str,
        mask_files: list[Path],
        axis_labels: list[str] | None = None,
        schema_name: str = "lsfm_cell_mapping.pointcloud_space",
        schema_version: str = "0.1.0",
    ) -> "PointCloudSpace":
        """Build metadata from a stack of mask files in image voxel space."""

        if axis_labels is None:
            axis_labels = ["x", "y", "z"]

        if len(mask_files) == 0:
            raise ValueError("Found no mask files, cannot build point-cloud space metadata")
        if len(resolution_um) != 3:
            raise ValueError(f"resolution_um must have length 3, got {resolution_um}")
        if len(axis_labels) != 3:
            raise ValueError(f"axis_labels must have length 3, got {axis_labels}")

        first_mask = tifffile.imread(mask_files[0])
        shape = [int(first_mask.shape[1]), int(first_mask.shape[0]), len(mask_files)]

        return cls(
            schema_name=schema_name,
            schema_version=schema_version,
            space_name=space_name,
            orientation=orientation,
            axis_labels=axis_labels,
            indexing=indexing,
            units="voxel",
            shape=shape,
            resolution_um=[float(value) for value in resolution_um],
        )


def build_pointcloud_space_metadata(
    *,
    space_name: str,
    orientation: str,
    resolution_um: list[float],
    indexing: str,
    mask_files: list[Path],
    axis_labels: list[str] | None = None,
    schema_name: str = "lsfm_cell_mapping.pointcloud_space",
    schema_version: str = "0.1.0",
) -> dict[str, Any]:
    """Build space-only metadata for a point cloud in image voxel space."""

    return PointCloudSpace.from_mask_files(
        space_name=space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        indexing=indexing,
        mask_files=mask_files,
        axis_labels=axis_labels,
        schema_name=schema_name,
        schema_version=schema_version,
    ).to_dict()


def write_pointcloud_space_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Write point-cloud space metadata to JSON."""

    PointCloudSpace.from_dict(metadata).to_json(output_path)
