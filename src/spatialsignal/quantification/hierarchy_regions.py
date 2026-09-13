"""Hierarchy-aware subject-space summaries for instance data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from atlaslevels import (
    AtlasIdMap,
    HierarchyBundle,
    load_preset_bundle,
    load_preset_id_map,
)

from spatialsignal.models import SpaceDefinition

from .atlas_regions import (
    SUPPORTED_REGION_ID_SPACES,
    _normalize_region_id,
    enrich_region_summary_with_atlas,
)
from .instance_regions import (
    OUT_OF_BOUNDS_REGION_ID,
    _summarize_objects_from_region_counts,
    _validate_same_sampling_space,
    _validate_volume_shape,
)


def summarize_objects_by_hierarchy_level(
    assigned_objects: pd.DataFrame,
    object_space: SpaceDefinition,
    annotation_data: np.ndarray,
    *,
    hierarchy_preset: str,
    hierarchy_level: str,
    region_id_space: str,
    annotation_space: SpaceDefinition | None = None,
    area_measurement_space: SpaceDefinition | None = None,
    id_map_preset: str = "allen_ccfv3_allen_to_kimlab16bit",
    region_column: str = "region_id",
    background_id: int = 0,
    out_of_bounds_id: int = OUT_OF_BOUNDS_REGION_ID,
    include_background: bool = False,
    include_out_of_bounds: bool = True,
) -> pd.DataFrame:
    """Summarize assigned objects at one curated atlas hierarchy level.

    Counts and annotation volumes are aggregated to hierarchy parents before
    density is calculated. Morphology statistics are calculated directly from
    all objects mapped to each parent rather than from detailed-region medians.
    """

    if region_column not in assigned_objects.columns:
        raise ValueError(f"Assigned object table is missing '{region_column}'")

    normalized_id_space = region_id_space.strip().lower()
    if normalized_id_space not in SUPPORTED_REGION_ID_SPACES:
        raise ValueError(
            f"region_id_space must be one of {sorted(SUPPORTED_REGION_ID_SPACES)}, "
            f"got {region_id_space!r}"
        )

    volume_space = annotation_space if annotation_space is not None else object_space
    if annotation_space is not None:
        _validate_same_sampling_space(
            object_space,
            annotation_space,
            volume_name="annotation",
        )
    _validate_volume_shape(annotation_data, volume_space, volume_name="annotation")

    bundle = load_preset_bundle(hierarchy_preset)
    bundle.get_level(hierarchy_level)
    kimlab_to_allen = (
        load_preset_id_map(id_map_preset).invert()
        if normalized_id_space == "kimlab16bit"
        else None
    )

    annotation_region_ids, annotation_region_voxels = np.unique(
        annotation_data,
        return_counts=True,
    )
    source_region_ids = {_normalize_region_id(value) for value in annotation_region_ids}
    source_region_ids.update(
        _normalize_region_id(value) for value in pd.unique(assigned_objects[region_column])
    )

    region_mapping = {
        source_region_id: _map_to_hierarchy_parent(
            source_region_id,
            bundle=bundle,
            hierarchy_level=hierarchy_level,
            kimlab_to_allen=kimlab_to_allen,
            background_id=background_id,
            out_of_bounds_id=out_of_bounds_id,
            region_id_space=normalized_id_space,
        )
        for source_region_id in source_region_ids
    }

    hierarchy_voxel_counts: dict[int, int] = {}
    for source_region_id, voxel_count in zip(
        annotation_region_ids,
        annotation_region_voxels,
        strict=True,
    ):
        normalized_region_id = _normalize_region_id(source_region_id)
        hierarchy_region_id = int(region_mapping[normalized_region_id])
        hierarchy_voxel_counts[hierarchy_region_id] = (
            hierarchy_voxel_counts.get(hierarchy_region_id, 0) + int(voxel_count)
        )

    grouped_region_ids = assigned_objects[region_column].map(region_mapping)
    area_space = area_measurement_space if area_measurement_space is not None else object_space
    summary = _summarize_objects_from_region_counts(
        assigned_objects,
        grouped_region_ids,
        hierarchy_voxel_counts,
        volume_space=volume_space,
        area_measurement_space=area_space,
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
        include_background=include_background,
        include_out_of_bounds=include_out_of_bounds,
    )
    summary = enrich_region_summary_with_atlas(
        summary,
        ontology_preset=bundle.atlas_name,
        region_id_space="allen",
        region_column="region_id",
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
    )
    summary = summary.drop(columns=["allen_region_id"])
    level_parent_ids = set(bundle.get_parent_ids(hierarchy_level))
    is_parent_residual = summary["region_id"].map(
        lambda region_id: (
            region_id not in {background_id, out_of_bounds_id}
            and region_id not in level_parent_ids
        )
    )
    summary.insert(
        summary.columns.get_loc("region_id") + 1,
        "is_parent_residual",
        is_parent_residual,
    )
    summary.insert(0, "hierarchy_preset", hierarchy_preset)
    return summary


def _map_to_hierarchy_parent(
    source_region_id: int | str,
    *,
    bundle: HierarchyBundle,
    hierarchy_level: str,
    kimlab_to_allen: AtlasIdMap | None,
    background_id: int,
    out_of_bounds_id: int,
    region_id_space: str,
) -> int | str:
    """Convert one source ID and map it to the selected hierarchy frontier."""

    if source_region_id in {background_id, out_of_bounds_id}:
        return source_region_id

    allen_region_id = (
        kimlab_to_allen.mapping.get(source_region_id)
        if kimlab_to_allen is not None
        else source_region_id
    )
    if allen_region_id not in bundle.ontology.nodes:
        raise ValueError(
            f"Region ID {source_region_id!r} is not valid in the {region_id_space!r} "
            f"namespace for hierarchy preset {bundle.family_name!r}"
        )

    parent_id = bundle.map_region_to_level_parent(allen_region_id, hierarchy_level)
    if parent_id is None:
        level_parent_ids = bundle.get_parent_ids(hierarchy_level)
        if any(
            bundle.ontology.is_descendant(level_parent_id, allen_region_id)
            for level_parent_id in level_parent_ids
        ):
            parent_id = allen_region_id
    if parent_id is None:
        raise ValueError(
            f"Region ID {source_region_id!r} does not map to hierarchy level "
            f"{hierarchy_level!r} in preset {bundle.family_name!r}"
        )
    return parent_id
