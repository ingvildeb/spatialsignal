"""High-level point-cloud building helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lsfm_cell_mapping.io.masks import find_mask_files
from lsfm_cell_mapping.pointcloud.centroids import extract_centroids_from_mask_stack
from lsfm_cell_mapping.qc import write_centroid_images


def build_pointcloud_from_masks(
    mask_dir: Path,
    out_dir: Path,
    *,
    pattern: str = "masks_*.tif*",
    slice_start: int = 1,
    one_based: bool = True,
    max_workers: int | None = 1,
    output_name: str = "pointcloud.csv",
    write_qc_images: bool = False,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> pd.DataFrame:
    """Build and export a canonical point cloud from a stack of mask images."""

    mask_files = find_mask_files(mask_dir, pattern=pattern)
    if show_progress:
        print(f"Found {len(mask_files)} mask slices in {mask_dir}")
    pointcloud = extract_centroids_from_mask_stack(
        mask_files,
        slice_start=slice_start,
        one_based=one_based,
        max_workers=max_workers,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pointcloud.to_csv(out_dir / output_name, index=False)
    if show_progress:
        print(f"Wrote point cloud CSV to {out_dir / output_name}")
    if write_qc_images:
        if show_progress:
            print("Writing centroid QC images")
        write_centroid_images(
            mask_files,
            pointcloud,
            out_dir,
            one_based=one_based,
        )
        if show_progress:
            print(f"Wrote centroid QC images to {out_dir}")
    return pointcloud
