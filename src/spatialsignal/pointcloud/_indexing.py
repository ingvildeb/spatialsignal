"""Helpers for indexing conventions used in point-cloud exports."""

from __future__ import annotations


VALID_INDEXING = {"zero_based", "one_based"}


def validate_indexing(indexing: str) -> str:
    """Validate and return a supported indexing mode."""

    if indexing not in VALID_INDEXING:
        raise ValueError(
            f"indexing must be one of {sorted(VALID_INDEXING)}, got {indexing}"
        )
    return indexing


def coordinate_offset(indexing: str) -> int:
    """Return the voxel-coordinate offset for the requested indexing mode."""

    indexing = validate_indexing(indexing)
    return 1 if indexing == "one_based" else 0


def default_slice_start(indexing: str) -> int:
    """Return the default starting slice index for an indexing convention."""

    return coordinate_offset(indexing)


def resolve_slice_start(indexing: str, slice_start: int | None = None) -> int:
    """Resolve a slice start, deriving the default from indexing when omitted."""

    indexing = validate_indexing(indexing)
    lower_bound = default_slice_start(indexing)
    resolved = lower_bound if slice_start is None else int(slice_start)
    if resolved < lower_bound:
        raise ValueError(
            f"slice_start must be >= {lower_bound} for {indexing} indexing, got {resolved}"
        )
    return resolved
