"""Point-cloud space metadata helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tifffile


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

    if axis_labels is None:
        axis_labels = ["slice", "row", "col"]

    if len(mask_files) == 0:
        raise ValueError("Found no mask files, cannot build point-cloud space metadata")
    if len(resolution_um) != 3:
        raise ValueError(f"resolution_um must have length 3, got {resolution_um}")
    if len(axis_labels) != 3:
        raise ValueError(f"axis_labels must have length 3, got {axis_labels}")

    first_mask = tifffile.imread(mask_files[0])
    shape = [len(mask_files), int(first_mask.shape[0]), int(first_mask.shape[1])]

    return {
        "schema_name": schema_name,
        "schema_version": schema_version,
        "space_name": space_name,
        "orientation": orientation,
        "axis_labels": axis_labels,
        "indexing": indexing,
        "units": "voxel",
        "shape": shape,
        "resolution_um": [float(value) for value in resolution_um],
    }


def write_pointcloud_space_metadata(metadata: dict[str, Any], output_path: Path) -> None:
    """Write point-cloud space metadata to JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
