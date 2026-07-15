"""Signal-point extraction for binary semantic segmentation masks."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

from spatialsignal.io.masks import assign_slices
from spatialsignal.pointcloud._indexing import (
    coordinate_offset,
    resolve_slice_start,
    validate_indexing,
)
from spatialsignal.utils.images import read_2d_mask


SIGNAL_POINT_REQUIRED_COLUMNS = ["point_id", "x", "y", "z"]


def extract_signal_points_from_mask(
    mask_path: Path,
    slice_index: int,
    *,
    indexing: str = "zero_based",
) -> pd.DataFrame:
    """Extract one signal-support point per nonzero pixel from a 2D mask image."""

    indexing = validate_indexing(indexing)

    mask = read_2d_mask(mask_path)
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return pd.DataFrame(columns=SIGNAL_POINT_REQUIRED_COLUMNS)

    offset = coordinate_offset(indexing)
    points = pd.DataFrame(
        {
            "point_id": np.arange(1, rows.size + 1, dtype=int),
            "x": cols.astype(int) + offset,
            "y": rows.astype(int) + offset,
            "z": np.full(rows.size, int(slice_index), dtype=int),
        },
        columns=SIGNAL_POINT_REQUIRED_COLUMNS,
    )
    return points


def _extract_signal_points_task(task: tuple[Path, int, str]) -> tuple[int, pd.DataFrame]:
    """Helper for parallel signal-point extraction."""

    mask_path, slice_index, indexing = task
    points = extract_signal_points_from_mask(mask_path, slice_index, indexing=indexing)
    return slice_index, points


def extract_signal_points_from_mask_stack(
    mask_files: list[Path],
    *,
    indexing: str = "zero_based",
    slice_start: int | None = None,
    max_workers: int | None = 1,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> pd.DataFrame:
    """Extract signal-support points from a naturally ordered mask stack."""

    indexing = validate_indexing(indexing)
    resolved_slice_start = resolve_slice_start(indexing, slice_start)
    assigned_masks = assign_slices(mask_files, slice_start=resolved_slice_start)
    tasks = [(mask_path, slice_index, indexing) for slice_index, mask_path in assigned_masks]

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
        point_tables_by_slice: dict[int, pd.DataFrame] = {}
        for completed, task in enumerate(tasks, start=1):
            slice_index, points = _extract_signal_points_task(task)
            point_tables_by_slice[slice_index] = points
            maybe_print_progress(completed)
    else:
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_extract_signal_points_task, task) for task in tasks]
                point_tables_by_slice = {}
                for completed, future in enumerate(as_completed(futures), start=1):
                    slice_index, points = future.result()
                    point_tables_by_slice[slice_index] = points
                    maybe_print_progress(completed)
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "Parallel signal-point extraction failed because a worker process exited unexpectedly. "
                "On Windows this commonly happens when running from an interactive cell or from a "
                "script without an `if __name__ == '__main__':` guard. Retry with `max_workers=1`, "
                "or run the workflow from a guarded script."
            ) from exc

    point_tables = [point_tables_by_slice[slice_index] for slice_index, _ in assigned_masks]
    if not point_tables:
        return pd.DataFrame(columns=SIGNAL_POINT_REQUIRED_COLUMNS)

    points = pd.concat(point_tables, ignore_index=True)
    points["point_id"] = np.arange(1, len(points) + 1, dtype=int)
    return points[SIGNAL_POINT_REQUIRED_COLUMNS]
