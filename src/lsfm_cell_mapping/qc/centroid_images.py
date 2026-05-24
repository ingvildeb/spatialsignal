"""Centroid image generation for slice-level quality control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def pointcloud_slice_to_image(
    slice_points: pd.DataFrame,
    image_shape: tuple[int, int],
    *,
    one_based: bool = True,
) -> np.ndarray:
    """Convert one slice of point-cloud rows into a binary centroid image."""

    image = np.zeros(image_shape, dtype=np.uint8)
    if slice_points.empty:
        return image

    x_offset = 1 if one_based else 0
    y_offset = 1 if one_based else 0

    ys = slice_points["y"].to_numpy(dtype=int) - y_offset
    xs = slice_points["x"].to_numpy(dtype=int) - x_offset

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
    one_based: bool = True,
    prefix: str = "centroids_",
) -> None:
    """Write one centroid QC image per mask slice."""

    out_dir.mkdir(parents=True, exist_ok=True)

    for slice_index, mask_path in enumerate(mask_files, start=1):
        mask = tifffile.imread(mask_path)
        slice_points = pointcloud.loc[pointcloud["z"] == slice_index]
        image = pointcloud_slice_to_image(
            slice_points,
            mask.shape,
            one_based=one_based,
        )
        mask_name = mask_path.name
        if mask_name.startswith("masks_"):
            mask_name = mask_name[len("masks_") :]
        tifffile.imwrite(out_dir / f"{prefix}{mask_name}", image, compression="lzw")
