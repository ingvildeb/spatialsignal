"""Voxelization utilities for spatial datasets."""

from .density import count_map_to_density_map
from .grids import (
    build_fraction_map_from_counts,
    compute_native_voxel_denominator_grid,
    empty_grid,
    filter_in_bounds_target_indices,
    map_indices_between_grids,
    pointcloud_to_zero_based_xyz,
)
from .nifti import build_nifti_ras_affine
from .spaces import make_subject_analysis_space
from .voxelize import (
    voxelize_point_centroids_to_count_map,
    voxelize_signal_masks_to_fraction_map,
    voxelize_to_space,
)

__all__ = [
    "build_fraction_map_from_counts",
    "build_nifti_ras_affine",
    "compute_native_voxel_denominator_grid",
    "count_map_to_density_map",
    "empty_grid",
    "filter_in_bounds_target_indices",
    "make_subject_analysis_space",
    "map_indices_between_grids",
    "pointcloud_to_zero_based_xyz",
    "voxelize_point_centroids_to_count_map",
    "voxelize_signal_masks_to_fraction_map",
    "voxelize_to_space",
]
