"""Core voxelization functions for spatial datasets."""

from __future__ import annotations

from typing import Any

import numpy as np

from lsfm_cell_mapping.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
    VoxelMap,
)


def empty_grid(
    space: SpaceDefinition,
    *,
    dtype: np.dtype = np.uint32,
) -> np.ndarray:
    """Allocate an empty voxel grid for a target space."""

    return np.zeros(tuple(space.shape), dtype=dtype)


def make_subject_analysis_space(
    source_space: SpaceDefinition,
    analysis_resolution_um: list[float],
    *,
    space_name: str = "subject_analysis_space",
) -> SpaceDefinition:
    """Derive a downsampled subject analysis space from a native source space."""

    if len(analysis_resolution_um) != 3:
        raise ValueError(
            "analysis_resolution_um must have length 3, got "
            f"{analysis_resolution_um}"
        )

    target_resolution = np.asarray(analysis_resolution_um, dtype=np.float64)
    if (target_resolution <= 0).any():
        raise ValueError(
            "analysis_resolution_um values must all be positive, got "
            f"{analysis_resolution_um}"
        )

    source_shape = np.asarray(source_space.shape, dtype=np.int64)
    source_resolution = np.asarray(source_space.resolution_um, dtype=np.float64)
    physical_extent_um = source_shape * source_resolution
    target_shape = np.ceil(physical_extent_um / target_resolution).astype(np.int64)

    return SpaceDefinition(
        space_name=space_name,
        orientation=source_space.orientation,
        axis_labels=list(source_space.axis_labels),
        indexing=source_space.indexing,
        units=source_space.units,
        shape=[int(value) for value in target_shape],
        resolution_um=[float(value) for value in target_resolution],
    )


def build_nifti_ras_affine(space: SpaceDefinition) -> np.ndarray:
    """Build a NIfTI-style RAS affine from BrainGlobe origin-based orientation.

    ``SpaceDefinition.orientation`` uses the BrainGlobe convention: each letter
    describes the anatomical side at voxel index 0 for the corresponding axis.
    This helper converts that origin-based convention into NIfTI's affine-based
    axis-direction convention in an RAS+ world.
    """

    if len(space.orientation) != 3:
        raise ValueError(
            f"orientation must have length 3, got {space.orientation}"
        )

    origin_letter_to_world = {
        "l": (0, 1.0),
        "r": (0, -1.0),
        "p": (1, 1.0),
        "a": (1, -1.0),
        "i": (2, 1.0),
        "s": (2, -1.0),
    }

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = 0.0
    for axis_index, (orient_letter, resolution_um) in enumerate(
        zip(
            space.orientation.lower(),
            space.resolution_um,
            strict=True,
        )
    ):
        if orient_letter not in origin_letter_to_world:
            raise ValueError(
                f"Unsupported orientation letter {orient_letter} in {space.orientation}"
            )
        world_axis, sign = origin_letter_to_world[orient_letter]
        affine[world_axis, axis_index] = sign * float(resolution_um)

    return affine


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


def voxelize_point_centroids_to_count_map(
    dataset: PointCloudDataset,
    target_space: SpaceDefinition,
    *,
    subject_name: str | None = None,
) -> VoxelMap:
    """Aggregate centroid points into a target voxel grid as counts."""

    _validate_centroid_pointcloud(dataset)
    dataset.validate_spatial_points()
    xyz_zero_based = pointcloud_to_zero_based_xyz(dataset)
    mapped_xyz = map_indices_between_grids(xyz_zero_based, dataset.space, target_space)
    kept_xyz, keep_mask = filter_in_bounds_target_indices(mapped_xyz, target_space)
    grid = accumulate_count_map(kept_xyz, target_space)

    metadata = DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=target_space,
        representation=DataRepresentation(
            kind="voxel_map",
            representation_type="count_map",
        ),
        processing=ProcessingProvenance(
            stage="voxelize_to_space",
            source_name=dataset.subject_name,
            parameters={
                "source_kind": dataset.metadata.representation.kind,
                "source_representation_type": dataset.metadata.representation.representation_type,
                "target_space_name": target_space.space_name,
            },
            summary={
                "input_points": int(len(dataset.points)),
                "kept_points": int(keep_mask.sum()),
                "dropped_points": int((~keep_mask).sum()),
                "nonzero_voxels": int(np.count_nonzero(grid)),
                "total_count": int(grid.sum()),
            },
        ),
    )

    return VoxelMap(
        subject_name=subject_name or dataset.subject_name,
        data=grid,
        metadata=metadata,
    )


def voxelize_to_space(
    dataset: PointCloudDataset,
    target_space: SpaceDefinition,
    *,
    subject_name: str | None = None,
) -> VoxelMap:
    """Voxelize a point-based dataset into a target space."""

    kind = dataset.metadata.representation.kind
    representation_type = dataset.metadata.representation.representation_type

    if kind == "point_cloud" and representation_type == "point_centroids":
        return voxelize_point_centroids_to_count_map(
            dataset,
            target_space,
            subject_name=subject_name,
        )

    raise NotImplementedError(
        "voxelize_to_space currently supports only "
        "kind='point_cloud' with representation_type='point_centroids'"
    )


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


def _validate_centroid_pointcloud(dataset: PointCloudDataset) -> None:
    """Check that a dataset is a centroid point cloud suitable for count-map voxelization."""

    representation = dataset.metadata.representation
    if representation.kind != "point_cloud":
        raise ValueError(f"Expected kind='point_cloud', got {representation.kind}")
    if representation.representation_type != "point_centroids":
        raise ValueError(
            "voxelize_point_centroids_to_count_map requires "
            "representation_type='point_centroids', got "
            f"{representation.representation_type}"
        )
