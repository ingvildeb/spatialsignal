"""High-level point-cloud building helpers."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd

from lsfm_cell_mapping.io.masks import find_mask_files
from lsfm_cell_mapping.pointcloud.centroids import extract_centroids_from_mask_stack
from lsfm_cell_mapping.pointcloud.signal_points import extract_signal_points_from_mask_stack
from lsfm_cell_mapping.models.metadata import DatasetMetadata
from lsfm_cell_mapping.qc import write_centroid_images


def make_subject_output_stem(subject_name: str) -> str:
    """Convert a subject name into a filesystem-friendly output stem."""

    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", subject_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("subject_name must contain at least one alphanumeric character")
    return normalized


def build_pointcloud_from_masks(
    mask_dir: Path,
    out_dir: Path,
    *,
    subject_name: str,
    space_name: str,
    orientation: str,
    resolution_um: list[float],
    representation_type: str = "point_centroids",
    pattern: str = "masks_*.tif*",
    slice_start: int = 1,
    one_based: bool = True,
    max_workers: int | None = 1,
    write_qc_images: bool = False,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> pd.DataFrame:
    """Build and export a canonical point cloud from a stack of mask images."""

    mask_files = find_mask_files(mask_dir, pattern=pattern)
    output_stem = make_subject_output_stem(subject_name)
    csv_path = out_dir / f"{output_stem}_pointcloud.csv"
    metadata_path = out_dir / f"{output_stem}_pointcloud_space.json"

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
    pointcloud.to_csv(csv_path, index=False)
    metadata = DatasetMetadata.from_mask_files(
        space_name=space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        indexing="one_based" if one_based else "zero_based",
        mask_files=mask_files,
        representation_type=representation_type,
    )
    metadata.to_json(metadata_path)
    if show_progress:
        print(f"Wrote point cloud CSV to {csv_path}")
        print(f"Wrote point cloud metadata to {metadata_path}")
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


def build_signal_points_from_masks(
    mask_dir: Path,
    out_dir: Path,
    *,
    subject_name: str,
    space_name: str,
    orientation: str,
    resolution_um: list[float],
    pattern: str = "*.tif*",
    slice_start: int = 1,
    one_based: bool = True,
    max_workers: int | None = 1,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> pd.DataFrame:
    """Build and export signal-support points from a stack of binary mask images."""

    mask_files = find_mask_files(mask_dir, pattern=pattern)
    output_stem = make_subject_output_stem(subject_name)
    csv_path = out_dir / f"{output_stem}_signal_points.csv"
    metadata_path = out_dir / f"{output_stem}_signal_points_space.json"

    if show_progress:
        print(f"Found {len(mask_files)} mask slices in {mask_dir}")
    signal_points = extract_signal_points_from_mask_stack(
        mask_files,
        slice_start=slice_start,
        one_based=one_based,
        max_workers=max_workers,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    signal_points.to_csv(csv_path, index=False)
    metadata = DatasetMetadata.from_mask_files(
        space_name=space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        indexing="one_based" if one_based else "zero_based",
        mask_files=mask_files,
        representation_type="signal_points",
    )
    metadata.to_json(metadata_path)
    if show_progress:
        print(f"Wrote signal-points CSV to {csv_path}")
        print(f"Wrote signal-points metadata to {metadata_path}")
    return signal_points
