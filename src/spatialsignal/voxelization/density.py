"""Conversions between discrete count maps and physical-density maps."""

from __future__ import annotations

import numpy as np

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    ProcessingProvenance,
    VoxelMap,
)


def count_map_to_density_map(count_map: VoxelMap) -> VoxelMap:
    """Convert object counts per voxel to object density per cubic millimeter."""

    representation = count_map.metadata.representation
    if representation.kind != "voxel_map" or representation.representation_type != "count_map":
        raise ValueError(
            "count_map_to_density_map requires a voxel_map with "
            "representation_type='count_map', got "
            f"kind={representation.kind!r}, representation_type="
            f"{representation.representation_type!r}"
        )

    resolution_um = np.asarray(count_map.space.resolution_um, dtype=np.float64)
    if resolution_um.shape != (3,) or not np.all(np.isfinite(resolution_um)):
        raise ValueError(
            "count-map resolution_um must contain three finite values, got "
            f"{count_map.space.resolution_um}"
        )
    if np.any(resolution_um <= 0):
        raise ValueError(
            "count-map resolution_um values must be positive, got "
            f"{count_map.space.resolution_um}"
        )
    if tuple(count_map.data.shape) != tuple(count_map.space.shape):
        raise ValueError(
            "Count-map array shape does not match its metadata space shape: "
            f"{count_map.data.shape} vs {tuple(count_map.space.shape)}"
        )
    if not np.issubdtype(count_map.data.dtype, np.number):
        raise ValueError(f"Count-map data must be numeric, got dtype {count_map.data.dtype}")
    if not np.all(np.isfinite(count_map.data)) or np.any(count_map.data < 0):
        raise ValueError("Count-map data must contain finite, non-negative values")

    voxel_volume_mm3 = float(np.prod(resolution_um) / 1_000_000_000.0)
    density = count_map.data.astype(np.float32) / np.float32(voxel_volume_mm3)
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=count_map.space,
        representation=DataRepresentation(
            kind="voxel_map",
            representation_type="density_map",
            value_units="objects_per_mm3",
        ),
        processing=ProcessingProvenance(
            stage="count_map_to_density_map",
            source_name=count_map.subject_name,
            parameters={
                "source_representation_type": representation.representation_type,
                "source_value_units": representation.value_units,
                "voxel_volume_mm3": voxel_volume_mm3,
            },
            summary={
                "total_count": float(count_map.data.sum()),
                "nonzero_voxels": int(np.count_nonzero(density)),
                "max_density_objects_per_mm3": (
                    float(density.max()) if density.size else 0.0
                ),
            },
        ),
    )
    return VoxelMap(
        subject_name=count_map.subject_name,
        data=density,
        metadata=metadata,
    )
