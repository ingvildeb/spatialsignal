"""Mask-discovery helpers for Cellpose output stacks."""

from __future__ import annotations

from pathlib import Path

from natsort import natsorted


def find_mask_files(
    mask_dir: Path,
    *,
    extensions: tuple[str, ...] = (".tif", ".tiff"),
    prefix: str | None = None,
    suffix: str | None = None,
) -> list[Path]:
    """Return naturally sorted mask files from a directory.

    Parameters
    ----------
    mask_dir:
        Directory containing Cellpose mask images.
    extensions:
        Accepted file extensions. Matching is case-insensitive.
    prefix:
        Optional filename prefix used to filter files.
    suffix:
        Optional filename suffix immediately before the file extension.

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

    normalized_extensions = _normalize_extensions(extensions)
    mask_files = natsorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in normalized_extensions
        and (prefix is None or path.stem.startswith(prefix))
        and (suffix is None or path.stem.endswith(suffix))
    )
    if not mask_files:
        filters = [f"extensions={tuple(sorted(normalized_extensions))!r}"]
        if prefix is not None:
            filters.append(f"prefix={prefix!r}")
        if suffix is not None:
            filters.append(f"suffix={suffix!r}")
        raise FileNotFoundError(
            f"No mask files matching {', '.join(filters)} were found in {mask_dir}"
        )

    return list(mask_files)


def _normalize_extensions(extensions: tuple[str, ...]) -> set[str]:
    """Validate and normalize literal file extensions."""

    if not extensions:
        raise ValueError("extensions must contain at least one file extension")

    normalized: set[str] = set()
    for extension in extensions:
        if not extension or any(character in extension for character in "*?[]"):
            raise ValueError(
                "extensions must contain literal file extensions without glob characters, "
                f"got {extension!r}"
            )
        normalized.add(extension.lower() if extension.startswith(".") else f".{extension.lower()}")
    return normalized


def assign_slices(mask_files: list[Path], slice_start: int = 0) -> list[tuple[int, Path]]:
    """Assign sequential slice indices to a sorted mask stack.

    This helper only enumerates files in order. Higher-level callers decide
    whether ``0`` or ``1`` is the meaningful default for a given indexing
    convention.
    """

    if slice_start < 0:
        raise ValueError(f"slice_start must be >= 0, got {slice_start}")

    return [
        (slice_index, mask_path)
        for slice_index, mask_path in enumerate(mask_files, start=slice_start)
    ]
