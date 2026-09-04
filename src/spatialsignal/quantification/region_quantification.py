"""High-level subject-space region quantification for cleaned objects."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from spatialsignal.integration.registration import LabelVolume
from spatialsignal.models import PointCloudDataset

from .atlas_regions import enrich_region_summary_with_atlas
from .hemispheres import HEMISPHERE_NAMES, validate_hemisphere_volume
from .instance_regions import (
    OUT_OF_BOUNDS_REGION_ID,
    _summarize_objects_from_region_counts,
    assign_objects_to_hemispheres,
    assign_objects_to_regions,
    remap_pointcloud_dataset_to_space,
    summarize_objects_by_region,
)


@dataclass(frozen=True)
class RegionQuantificationResult:
    """Outputs from assigning cleaned objects to subject-space atlas regions."""

    remapped_objects: PointCloudDataset
    assigned_objects: pd.DataFrame
    region_summary: pd.DataFrame
    summary_csv: Path | None = None
    qc_png: Path | None = None


def quantify_objects_by_region(
    objects: PointCloudDataset,
    annotation: LabelVolume,
    *,
    hemisphere: LabelVolume | None = None,
    ontology_preset: str = "allen_ccfv3",
    region_id_space: str = "allen",
    include_background: bool = False,
    include_out_of_bounds: bool = True,
    background_id: int = 0,
    out_of_bounds_id: int = OUT_OF_BOUNDS_REGION_ID,
    output_csv: Path | None = None,
    output_qc: Path | None = None,
    generate_qc: bool = True,
) -> RegionQuantificationResult:
    """Assign cleaned objects to atlas regions and build a named regional report.

    The annotation must already be transformed into the same physical subject
    space as ``objects``. Object coordinates are remapped into the annotation's
    sampling grid before its region IDs are sampled. Annotation background is
    sufficient to identify non-region signal; a separate brain mask is not
    required. When a hemisphere map is supplied, it must use ``1 = left`` and
    ``2 = right`` at every annotated voxel. The report then contains explicit
    bilateral, left, and right metric columns.
    """

    representation = objects.metadata.representation
    if not (
        representation.kind == "point_cloud"
        and representation.representation_type == "cleaned_objects"
    ):
        raise ValueError(
            "quantify_objects_by_region requires a cleaned-object point cloud; got "
            f"kind={representation.kind!r}, "
            f"representation_type={representation.representation_type!r}"
        )

    summary_csv = Path(output_csv) if output_csv is not None else None
    if summary_csv is not None and summary_csv.suffix.lower() != ".csv":
        raise ValueError(f"output_csv must have a .csv suffix, got {summary_csv}")
    qc_png = Path(output_qc) if output_qc is not None else None
    if qc_png is not None and qc_png.suffix.lower() != ".png":
        raise ValueError(f"output_qc must have a .png suffix, got {qc_png}")
    if qc_png is not None and not generate_qc:
        raise ValueError("output_qc cannot be provided when generate_qc is False")
    if generate_qc and qc_png is None and summary_csv is not None:
        qc_png = _default_qc_path(summary_csv)

    remapped_objects = remap_pointcloud_dataset_to_space(
        objects,
        annotation.space,
    )
    assigned_objects = assign_objects_to_regions(
        remapped_objects.points,
        remapped_objects.space,
        annotation.data,
        annotation_space=annotation.space,
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
    )
    if hemisphere is not None:
        validate_hemisphere_volume(
            hemisphere,
            annotation,
            background_id=background_id,
        )
        assigned_objects = assign_objects_to_hemispheres(
            assigned_objects,
            remapped_objects.space,
            hemisphere.data,
        )

    region_summary = _summarize_object_scopes(
        assigned_objects,
        remapped_objects,
        objects,
        annotation,
        hemisphere=hemisphere,
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
        include_background=include_background,
        include_out_of_bounds=include_out_of_bounds,
    )
    region_summary = enrich_region_summary_with_atlas(
        region_summary,
        ontology_preset=ontology_preset,
        region_id_space=region_id_space,
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
    )

    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        region_summary.to_csv(summary_csv, index=False)

    result = RegionQuantificationResult(
        remapped_objects=remapped_objects,
        assigned_objects=assigned_objects,
        region_summary=region_summary,
        summary_csv=summary_csv,
    )
    if qc_png is not None:
        from .qc import write_region_quantification_qc

        write_region_quantification_qc(result, annotation, qc_png)
        result = replace(result, qc_png=qc_png)
    return result


def _summarize_object_scopes(
    assigned_objects: pd.DataFrame,
    remapped_objects: PointCloudDataset,
    source_objects: PointCloudDataset,
    annotation: LabelVolume,
    *,
    hemisphere: LabelVolume | None,
    background_id: int,
    out_of_bounds_id: int,
    include_background: bool,
    include_out_of_bounds: bool,
) -> pd.DataFrame:
    """Build consistent bilateral and optional lateralized object metrics."""

    bilateral = summarize_objects_by_region(
        assigned_objects,
        remapped_objects.space,
        annotation.data,
        annotation_space=annotation.space,
        area_measurement_space=source_objects.space,
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
        include_background=include_background,
        include_out_of_bounds=include_out_of_bounds,
    )
    summary = _prefix_scope_columns(bilateral, "bilateral")
    if hemisphere is None:
        return summary

    all_region_ids = [int(value) for value in np.unique(annotation.data)]
    voxel_volume_space = annotation.space
    for hemisphere_id, hemisphere_name in HEMISPHERE_NAMES.items():
        scope_voxels = np.asarray(hemisphere.data) == hemisphere_id
        region_counts = {
            region_id: int(
                np.count_nonzero(
                    (np.asarray(annotation.data) == region_id) & scope_voxels
                )
            )
            for region_id in all_region_ids
        }
        scope_objects = assigned_objects.loc[
            assigned_objects["hemisphere_id"] == hemisphere_id
        ]
        scope_summary = _summarize_objects_from_region_counts(
            scope_objects,
            scope_objects["region_id"],
            region_counts,
            volume_space=voxel_volume_space,
            area_measurement_space=source_objects.space,
            background_id=background_id,
            out_of_bounds_id=out_of_bounds_id,
            include_background=include_background,
            include_out_of_bounds=False,
        )
        scope_summary = _prefix_scope_columns(scope_summary, hemisphere_name)
        summary = summary.merge(
            scope_summary,
            on="region_id",
            how="left",
            validate="one_to_one",
        )
    return summary


def _prefix_scope_columns(summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Prefix all regional metrics while retaining the shared region key."""

    return summary.rename(
        columns={
            column: f"{scope}_{column}"
            for column in summary.columns
            if column != "region_id"
        }
    )


def _default_qc_path(summary_csv: Path) -> Path:
    """Derive the canonical QC filename from a regional-summary CSV path."""

    stem = summary_csv.stem
    suffix = "_region_summary"
    output_stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return summary_csv.with_name(f"{output_stem}_quantification_qc.png")
