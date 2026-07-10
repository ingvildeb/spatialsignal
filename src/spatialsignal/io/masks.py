"""Mask-discovery helpers for Cellpose output stacks."""

from __future__ import annotations

from pathlib import Path

from natsort import natsorted


def find_mask_files(mask_dir: Path, pattern: str = "masks_*.tif*") -> list[Path]:
    """Return naturally sorted mask files from a directory.

    Parameters
    ----------
    mask_dir:
        Directory containing Cellpose mask images.
    pattern:
        Glob pattern used to identify mask files.

    Returns
    -------
    list[Path]
        Naturally sorted list of matching files.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or no matching files are found.
    NotADirectoryError
        If ``mask_dir`` is not a directory.
    """

    if not mask_dir.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask path is not a directory: {mask_dir}")

    mask_files = natsorted(mask_dir.glob(pattern))
    if not mask_files:
        raise FileNotFoundError(
            f"No mask files matching pattern '{pattern}' were found in {mask_dir}"
        )

    return list(mask_files)


def assign_slices(mask_files: list[Path], slice_start: int = 1) -> list[tuple[int, Path]]:
    """Assign sequential slice indices to a sorted mask stack."""

    if slice_start < 1:
        raise ValueError(f"slice_start must be >= 1, got {slice_start}")

    return [(slice_index, mask_path) for slice_index, mask_path in enumerate(mask_files, start=slice_start)]
