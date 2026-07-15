"""Representation-aware voxelization functions for spatial point datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
    VoxelMap,
)
from spatialsignal.utils.images import read_2d_mask
from .grids import (
    accumulate_count_map,
    build_fraction_map_from_counts,
    compute_native_voxel_denominator_grid,
    filter_in_bounds_target_indices,
    map_indices_between_grids,
    pointcloud_to_zero_based_xyz,
)


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
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=target_space,
        representation=DataRepresentation(
            kind="voxel_map",
            representation_type="count_map",
            value_units="objects_per_voxel",
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

    if kind == "point_cloud" and representation_type in {
        "point_centroids",
        "cleaned_objects",
    }:
        return voxelize_point_centroids_to_count_map(
            dataset,
            target_space,
            subject_name=subject_name,
        )

    raise NotImplementedError(
        "voxelize_to_space currently supports kind='point_cloud' with "
        "representation_type='point_centroids' or 'cleaned_objects'"
    )


def voxelize_signal_masks_to_fraction_map(
    mask_files: list[Path],
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
    *,
    subject_name: str,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> VoxelMap:
    """Voxelize binary semantic segmentation masks directly into a fraction map."""

    if progress_interval < 1:
        raise ValueError(f"progress_interval must be >= 1, got {progress_interval}")

    dataset = _accumulate_signal_masks_to_fraction_map(
        mask_files=mask_files,
        source_space=source_space,
        target_space=target_space,
        subject_name=subject_name,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )
    return dataset


def _accumulate_signal_masks_to_fraction_map(
    mask_files: list[Path],
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
    *,
    subject_name: str,
    show_progress: bool,
    progress_interval: int,
) -> VoxelMap:
    _validate_compatible_point_spaces_for_fraction_map(source_space, target_space)

    numerator_grid, input_signal_points = accumulate_signal_masks_to_target_counts(
        mask_files=mask_files,
        source_space=source_space,
        target_space=target_space,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )
    denominator_grid = compute_native_voxel_denominator_grid(source_space, target_space)
    fraction_grid = build_fraction_map_from_counts(numerator_grid, denominator_grid)

    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=target_space,
        representation=DataRepresentation(
            kind="voxel_map",
            representation_type="fraction_map",
        ),
        processing=ProcessingProvenance(
            stage="voxelize_signal_masks_to_fraction_map",
            source_name=subject_name,
            parameters={
                "source_kind": "point_cloud",
                "source_representation_type": "signal_points",
                "target_space_name": target_space.space_name,
            },
            summary={
                "input_signal_points": int(input_signal_points),
                "nonzero_voxels": int(np.count_nonzero(fraction_grid)),
                "sum_signal_counts": int(numerator_grid.sum()),
                "max_fraction": float(fraction_grid.max()) if fraction_grid.size else 0.0,
            },
        ),
    )
    return VoxelMap(subject_name=subject_name, data=fraction_grid, metadata=metadata)


def accumulate_signal_masks_to_target_counts(
    mask_files: list[Path],
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
    *,
    show_progress: bool = False,
    progress_interval: int = 25,
) -> tuple[np.ndarray, int]:
    """Stream binary masks and accumulate signal-support counts in target voxels."""

    _validate_compatible_point_spaces_for_fraction_map(source_space, target_space)
    if len(mask_files) != int(source_space.shape[2]):
        raise ValueError(
            "Number of mask files must match source_space z dimension, got "
            f"{len(mask_files)} files and shape[2]={source_space.shape[2]}"
        )

    offset = 1 if source_space.indexing == "one_based" else 0
    total_files = len(mask_files)
    numerator_grid = np.zeros(tuple(target_space.shape), dtype=np.uint32)
    signal_points = 0

    source_resolution = np.asarray(source_space.resolution_um, dtype=np.float64)
    target_resolution = np.asarray(target_space.resolution_um, dtype=np.float64)

    for slice_zero_based, mask_path in enumerate(mask_files):
        mask = read_2d_mask(mask_path)
        rows, cols = np.nonzero(mask)
        signal_points += int(rows.size)
        if rows.size:
            z_value = slice_zero_based + offset
            xyz = np.column_stack(
                (
                    cols.astype(np.int64) + offset,
                    rows.astype(np.int64) + offset,
                    np.full(rows.size, z_value, dtype=np.int64),
                )
            )
            if source_space.indexing == "one_based":
                xyz -= 1
            physical_um = (xyz.astype(np.float64) + 0.5) * source_resolution
            mapped = np.floor(physical_um / target_resolution).astype(np.int64)
            kept_xyz, _ = filter_in_bounds_target_indices(mapped, target_space)
            numerator_grid += accumulate_count_map(kept_xyz, target_space)

        if show_progress and (
            (slice_zero_based + 1) % progress_interval == 0
            or slice_zero_based + 1 == total_files
        ):
            print(f"Processed {slice_zero_based + 1}/{total_files} mask slices")

    return numerator_grid, signal_points


def _validate_centroid_pointcloud(dataset: PointCloudDataset) -> None:
    """Check that a dataset is a centroid point cloud suitable for count-map voxelization."""

    representation = dataset.metadata.representation
    if representation.kind != "point_cloud":
        raise ValueError(f"Expected kind='point_cloud', got {representation.kind}")
    if representation.representation_type not in {"point_centroids", "cleaned_objects"}:
        raise ValueError(
            "voxelize_point_centroids_to_count_map requires "
            "representation_type='point_centroids' or 'cleaned_objects', got "
            f"{representation.representation_type}"
        )


def _validate_compatible_point_spaces_for_fraction_map(
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
) -> None:
    """Check that source and target spaces can be directly aggregated."""

    if source_space.axis_labels != target_space.axis_labels:
        raise ValueError(
            f"Source and target axis_labels must match, got {source_space.axis_labels} and {target_space.axis_labels}"
        )
    if source_space.orientation != target_space.orientation:
        raise ValueError(
            f"Source and target orientation must match, got {source_space.orientation} and {target_space.orientation}"
        )
