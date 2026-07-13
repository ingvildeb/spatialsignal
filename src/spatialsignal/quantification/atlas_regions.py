"""Atlas metadata enrichment for region-level reports."""

from __future__ import annotations

from typing import Any

import pandas as pd

from atlaslevels import load_preset_id_map, load_preset_ontology


SUPPORTED_REGION_ID_SPACES = {"allen", "kimlab16bit"}


def enrich_region_summary_with_atlas(
    region_summary: pd.DataFrame,
    *,
    ontology_preset: str = "allen_ccfv3",
    region_id_space: str = "allen",
    id_map_preset: str = "allen_ccfv3_allen_to_kimlab16bit",
    region_column: str = "region_id",
    background_id: int = 0,
    out_of_bounds_id: int = -1,
    strict: bool = True,
) -> pd.DataFrame:
    """Attach canonical atlas IDs, names, acronyms, and colors to a region report."""

    if region_column not in region_summary.columns:
        raise ValueError(f"Region summary is missing '{region_column}'")
    normalized_id_space = region_id_space.strip().lower()
    if normalized_id_space not in SUPPORTED_REGION_ID_SPACES:
        raise ValueError(
            f"region_id_space must be one of {sorted(SUPPORTED_REGION_ID_SPACES)}, "
            f"got {region_id_space!r}"
        )

    ontology = load_preset_ontology(ontology_preset)
    kimlab_to_allen = None
    if normalized_id_space == "kimlab16bit":
        kimlab_to_allen = load_preset_id_map(id_map_preset).invert()

    atlas_rows: list[dict[str, Any]] = []
    unknown_region_ids: list[int | str] = []
    for raw_region_id in region_summary[region_column]:
        region_id = _normalize_region_id(raw_region_id)
        if region_id == background_id:
            atlas_rows.append(_special_region_metadata("Background", "background", "#000000"))
            continue
        if region_id == out_of_bounds_id:
            atlas_rows.append(
                _special_region_metadata("Out of bounds", "out_of_bounds", "#808080")
            )
            continue

        allen_region_id = (
            kimlab_to_allen.mapping.get(region_id)
            if kimlab_to_allen is not None
            else region_id
        )
        if allen_region_id not in ontology.nodes:
            unknown_region_ids.append(region_id)
            atlas_rows.append(_unknown_region_metadata())
            continue

        node = ontology.get_node(allen_region_id)
        atlas_rows.append(
            {
                "allen_region_id": node.id,
                "region_acronym": node.acronym,
                "region_name": node.name,
                "region_color": node.color,
            }
        )

    if unknown_region_ids and strict:
        unknown_values = sorted(set(unknown_region_ids), key=str)
        raise ValueError(
            f"Region IDs are not valid in the {region_id_space!r} namespace for "
            f"ontology preset {ontology_preset!r}: {unknown_values}"
        )

    enriched = region_summary.copy()
    atlas_metadata = pd.DataFrame(atlas_rows, index=enriched.index)
    insert_at = enriched.columns.get_loc(region_column) + 1
    for column in reversed(list(atlas_metadata.columns)):
        enriched.insert(insert_at, column, atlas_metadata[column])
    return enriched


def _normalize_region_id(value: Any) -> int | str:
    """Normalize integer-like table values without changing string IDs."""

    if pd.isna(value):
        raise ValueError("Region summary contains a missing region ID")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _special_region_metadata(name: str, acronym: str, color: str) -> dict[str, Any]:
    """Build metadata for a non-ontology summary row."""

    return {
        "allen_region_id": pd.NA,
        "region_acronym": acronym,
        "region_name": name,
        "region_color": color,
    }


def _unknown_region_metadata() -> dict[str, Any]:
    """Build placeholder metadata used when non-strict enrichment is requested."""

    return {
        "allen_region_id": pd.NA,
        "region_acronym": "unknown",
        "region_name": "Unknown region",
        "region_color": pd.NA,
    }
