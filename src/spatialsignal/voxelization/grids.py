"""Low-level voxel-grid allocation and index-mapping helpers."""

from __future__ import annotations

import numpy as np

from spatialsignal.models import PointCloudDataset, SpaceDefinition


def empty_grid(
    space: SpaceDefinition,
    *,
    dtype: np.dtype = np.uint32,
) -> np.ndarray:
    """Allocate an empty voxel grid for a target space."""

    return np.zeros(tuple(space.shape), dtype=dtype)


def pointcloud_to_zero_based_xyz(
    dataset: PointCloudDataset,
) -> np.ndarray:
    """Return point coordinates as an (N, 3) zero-based xyz integer array."""

    xyz = dataset.points[dataset.space.axis_labels].to_numpy(dtype=np.int64, copy=True)
    if dataset.space.indexing == "one_based":
        xyz -= 1
    return xyz


def map_indices_between_grids(
    xyz_zero_based: np.ndarray,
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> np.ndarray:
    """Map zero-based xyz indices from a source grid to a target grid."""

    _validate_compatible_grid_axes(source_space, target_space)
    source_resolution = np.asarray(source_space.resolution_um, dtype=np.float64)
    target_resolution = np.asarray(target_space.resolution_um, dtype=np.float64)

    physical_um = (xyz_zero_based.astype(np.float64) + 0.5) * source_resolution
    return np.floor(physical_um / target_resolution).astype(np.int64)


def filter_in_bounds_target_indices(
    xyz_zero_based: np.ndarray,
    target_space: SpaceDefinition,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only indices that fall within the target grid bounds."""

    shape = np.asarray(target_space.shape, dtype=np.int64)
    keep_mask = np.all((xyz_zero_based >= 0) & (xyz_zero_based < shape), axis=1)
    return xyz_zero_based[keep_mask], keep_mask


def accumulate_count_map(
    xyz_zero_based: np.ndarray,
    target_space: SpaceDefinition,
) -> np.ndarray:
    """Accumulate per-point counts into a target voxel grid."""

    grid = empty_grid(target_space, dtype=np.uint32)
    if xyz_zero_based.size == 0:
        return grid

    np.add.at(
        grid,
        (xyz_zero_based[:, 0], xyz_zero_based[:, 1], xyz_zero_based[:, 2]),
        1,
    )
    return grid


def compute_native_voxel_denominator_grid(
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> np.ndarray:
    """Count how many native voxels contribute to each target voxel."""

    _validate_compatible_grid_axes(source_space, target_space)
    per_axis_counts = []
    for axis_size, source_resolution, target_resolution, target_axis_size in zip(
        source_space.shape,
        source_space.resolution_um,
        target_space.resolution_um,
        target_space.shape,
        strict=True,
    ):
        source_indices = np.arange(int(axis_size), dtype=np.float64)
        mapped = np.floor(
            ((source_indices + 0.5) * float(source_resolution)) / float(target_resolution)
        ).astype(np.int64)
        counts = np.bincount(mapped, minlength=int(target_axis_size)).astype(np.uint32)
        if len(counts) > int(target_axis_size):
            counts = counts[: int(target_axis_size)]
        per_axis_counts.append(counts)

    x_counts, y_counts, z_counts = per_axis_counts
    denominator = (
        x_counts[:, np.newaxis, np.newaxis]
        * y_counts[np.newaxis, :, np.newaxis]
        * z_counts[np.newaxis, np.newaxis, :]
    )
    return denominator.astype(np.uint32, copy=False)


def build_fraction_map_from_counts(
    signal_count_grid: np.ndarray,
    denominator_grid: np.ndarray,
) -> np.ndarray:
    """Convert signal-support counts and denominator counts into a fraction map."""

    if signal_count_grid.shape != denominator_grid.shape:
        raise ValueError(
            "signal_count_grid and denominator_grid must have matching shapes, got "
            f"{signal_count_grid.shape} and {denominator_grid.shape}"
        )

    fraction = np.zeros(signal_count_grid.shape, dtype=np.float32)
    np.divide(
        signal_count_grid.astype(np.float32, copy=False),
        denominator_grid.astype(np.float32, copy=False),
        out=fraction,
        where=denominator_grid != 0,
    )
    return fraction


def _validate_compatible_grid_axes(
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> None:
    """Check that source and target grids are compatible for direct aggregation."""

    if source_space.axis_labels != target_space.axis_labels:
        raise ValueError(
            f"Source and target axis_labels must match, got {source_space.axis_labels} and {target_space.axis_labels}"
        )
    if source_space.orientation != target_space.orientation:
        raise ValueError(
            f"Source and target orientation must match, got {source_space.orientation} and {target_space.orientation}"
        )
