"""Centroid extraction for Cellpose mask images."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from lsfm_cell_mapping.io.masks import assign_slices


POINTCLOUD_COLUMNS = ["seg_num", "row", "col", "slice"]


def matlab_round(value: float) -> int:
    """Round positive values the way MATLAB does for centroid coordinates.

    For positive values, MATLAB rounds halves away from zero, while Python's
    built-in ``round`` uses bankers rounding. Image coordinates here are
    non-negative, so ``floor(value + 0.5)`` matches the legacy behavior.
    """

    return int(np.floor(value + 0.5))


def extract_centroids_from_mask(
    mask_path: Path,
    slice_index: int,
    *,
    one_based: bool = True,
) -> pd.DataFrame:
    """Extract one centroid per labeled object from a 2D mask image.

    Parameters
    ----------
    mask_path:
        Path to a single 2D label image written by Cellpose.
    slice_index:
        Sequential slice number assigned to the image in stack order.
    one_based:
        If True, export integer centroid coordinates as 1-based indices to
        match the legacy MATLAB workflow.
    """

    mask = tifffile.imread(mask_path)
    labels = np.unique(mask)
    labels = labels[labels != 0]

    if labels.size == 0:
        return pd.DataFrame(columns=POINTCLOUD_COLUMNS)

    centroids = ndimage.center_of_mass(mask, labels=mask, index=labels)
    offset = 1 if one_based else 0

    rows: list[dict[str, int]] = []
    for seg_num, (row, col) in zip(labels, centroids, strict=True):
        rows.append(
            {
                "seg_num": int(seg_num),
                "row": matlab_round(float(row)) + offset,
                "col": matlab_round(float(col)) + offset,
                "slice": int(slice_index),
            }
        )

    return pd.DataFrame(rows, columns=POINTCLOUD_COLUMNS)


def _extract_centroids_task(task: tuple[Path, int, bool]) -> tuple[int, pd.DataFrame]:
    """Helper for parallel centroid extraction."""

    mask_path, slice_index, one_based = task
    pointcloud = extract_centroids_from_mask(mask_path, slice_index, one_based=one_based)
    return slice_index, pointcloud


def extract_centroids_from_mask_stack(
    mask_files: list[Path],
    *,
    slice_start: int = 1,
    one_based: bool = True,
    max_workers: int | None = 1,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> pd.DataFrame:
    """Extract centroids from a naturally ordered mask stack."""

    assigned_masks = assign_slices(mask_files, slice_start=slice_start)
    tasks = [(mask_path, slice_index, one_based) for slice_index, mask_path in assigned_masks]

    if max_workers in (None, 0):
        max_workers = 1
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if progress_interval < 1:
        raise ValueError(f"progress_interval must be >= 1, got {progress_interval}")

    total_tasks = len(tasks)

    def maybe_print_progress(completed: int) -> None:
        if not show_progress:
            return
        if completed % progress_interval == 0 or completed == total_tasks:
            print(f"Processed {completed}/{total_tasks} mask slices")

    if max_workers == 1:
        pointcloud_tables_by_slice: dict[int, pd.DataFrame] = {}
        for completed, task in enumerate(tasks, start=1):
            slice_index, pointcloud = _extract_centroids_task(task)
            pointcloud_tables_by_slice[slice_index] = pointcloud
            maybe_print_progress(completed)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_extract_centroids_task, task) for task in tasks]
            pointcloud_tables_by_slice = {}
            for completed, future in enumerate(as_completed(futures), start=1):
                slice_index, pointcloud = future.result()
                pointcloud_tables_by_slice[slice_index] = pointcloud
                maybe_print_progress(completed)

    pointcloud_tables = [
        pointcloud_tables_by_slice[slice_index]
        for slice_index, _ in assigned_masks
    ]

    if not pointcloud_tables:
        return pd.DataFrame(columns=POINTCLOUD_COLUMNS)

    return pd.concat(pointcloud_tables, ignore_index=True)
