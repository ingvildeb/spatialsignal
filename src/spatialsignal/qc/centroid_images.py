"""Centroid image generation for slice-level quality control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from spatialsignal.io.masks import assign_slices
from spatialsignal.pointcloud._indexing import (
    coordinate_offset,
    resolve_slice_start,
    validate_indexing,
)


def pointcloud_slice_to_image(
    slice_points: pd.DataFrame,
    image_shape: tuple[int, int],
    *,
    indexing: str = "zero_based",
) -> np.ndarray:
    """Convert one slice of point-cloud rows into a binary centroid image."""

    indexing = validate_indexing(indexing)

    image = np.zeros(image_shape, dtype=np.uint8)
    if slice_points.empty:
        return image

    offset = coordinate_offset(indexing)
    ys = slice_points["y"].to_numpy(dtype=int) - offset
    xs = slice_points["x"].to_numpy(dtype=int) - offset

    in_bounds = (
        (ys >= 0)
        & (ys < image_shape[0])
        & (xs >= 0)
        & (xs < image_shape[1])
    )
    image[ys[in_bounds], xs[in_bounds]] = 255
    return image


def write_centroid_images(
    mask_files: list[Path],
    pointcloud: pd.DataFrame,
    out_dir: Path,
    *,
    indexing: str = "zero_based",
    slice_start: int | None = None,
    prefix: str = "centroids_",
) -> None:
    """Write one centroid QC image per mask slice."""

    indexing = validate_indexing(indexing)
    resolved_slice_start = resolve_slice_start(indexing, slice_start)

    out_dir.mkdir(parents=True, exist_ok=True)

    for slice_index, mask_path in assign_slices(mask_files, slice_start=resolved_slice_start):
        mask = tifffile.imread(mask_path)
        slice_points = pointcloud.loc[pointcloud["z"] == slice_index]
        image = pointcloud_slice_to_image(
            slice_points,
            mask.shape,
            indexing=indexing,
        )
        mask_name = mask_path.name
        if mask_name.startswith("masks_"):
            mask_name = mask_name[len("masks_") :]
        tifffile.imwrite(out_dir / f"{prefix}{mask_name}", image, compression="lzw")
