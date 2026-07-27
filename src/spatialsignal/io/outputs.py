"""Reusable output writers for spatialsignal datasets."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
    VoxelMap,
)
from spatialsignal.voxelization.nifti import build_nifti_ras_affine

if TYPE_CHECKING:
    from spatialsignal.integration import LabelVolume
    from spatialsignal.pointcloud.colocalization import (
        ColocalizationResult,
        ObjectRelationshipResult,
    )
    from spatialsignal.pointcloud.deduplicate import DeduplicationResult


@dataclass(frozen=True)
class VoxelMapOutputPaths:
    """Paths written for a saved voxel map."""

    array_path: Path | None
    metadata_path: Path
    nifti_path: Path | None
    nifti_written: bool


@dataclass(frozen=True)
class DeduplicationOutputPaths:
    """Paths written for saved deduplication outputs."""

    objects_table: Path
    objects_json: Path
    membership_table: Path
    edges_table: Path | None


@dataclass(frozen=True)
class ColocalizationOutputPaths:
    """Paths written for saved colocalization evidence and QC."""

    matches_table: Path
    relationships_table: Path | None
    metadata_path: Path
    qc_report: Path | None
    qc_images_dir: Path | None


@dataclass(frozen=True)
class InstanceRegionQuantificationOutputPaths:
    """Paths written for instance region-assignment outputs."""

    assigned_objects_table: Path
    assigned_objects_json: Path
    region_summary_csv: Path
    qc_png: Path | None


def make_subject_output_stem(subject_name: str) -> str:
    """Convert a subject name into a filesystem-friendly output stem."""

    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", subject_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("subject_name must contain at least one alphanumeric character")
    return normalized


def write_nifti_voxel_map(
    data_xyz: np.ndarray,
    space: SpaceDefinition,
    output_path: Path,
) -> bool:
    """Write a voxel map as NIfTI if nibabel is available."""

    try:
        import nibabel as nib
    except ModuleNotFoundError:
        return False

    affine = build_nifti_ras_affine(space)
    image = nib.Nifti1Image(data_xyz, affine)
    image.header.set_xyzt_units(xyz="mm")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return True


def save_voxel_map_outputs(
    voxel_map: VoxelMap,
    out_dir: Path,
    *,
    name_suffix: str,
    output_stem: str | None = None,
    formats: tuple[str, ...] = ("npy", "nifti"),
) -> VoxelMapOutputPaths:
    """Save a voxel map in selected formats plus a metadata JSON sidecar."""

    requested_formats = set(formats)
    if not requested_formats:
        raise ValueError("formats must contain at least one output format")
    unsupported_formats = requested_formats.difference({"npy", "nifti"})
    if unsupported_formats:
        raise ValueError(f"Unsupported voxel-map output formats: {sorted(unsupported_formats)}")

    if output_stem is None:
        output_stem = make_subject_output_stem(voxel_map.subject_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{output_stem}_{name_suffix}"
    array_path = base_path.with_suffix(".npy") if "npy" in requested_formats else None
    nifti_path = (
        out_dir / f"{output_stem}_{name_suffix}.nii.gz"
        if "nifti" in requested_formats
        else None
    )
    metadata_path = out_dir / f"{output_stem}_{name_suffix}_space.json"

    if array_path is not None:
        np.save(array_path, voxel_map.data)
    voxel_map.metadata.to_json(metadata_path)
    nifti_written = (
        write_nifti_voxel_map(voxel_map.data, voxel_map.space, nifti_path)
        if nifti_path is not None
        else False
    )

    return VoxelMapOutputPaths(
        array_path=array_path,
        metadata_path=metadata_path,
        nifti_path=nifti_path,
        nifti_written=nifti_written,
    )


def save_deduplication_outputs(
    dataset: PointCloudDataset,
    result: DeduplicationResult,
    out_dir: Path,
    *,
    source_name: str | None = None,
    parameters: dict[str, Any] | None = None,
    output_stem: str | None = None,
    write_edge_table: bool = False,
) -> DeduplicationOutputPaths:
    """Save cleaned-object tables and provenance for deduplication outputs."""

    from spatialsignal.pointcloud.deduplicate import summarize_deduplication_result

    if output_stem is None:
        output_stem = make_subject_output_stem(dataset.subject_name)
    if source_name is None:
        source_name = f"{output_stem}_pointcloud.parquet"

    out_dir.mkdir(parents=True, exist_ok=True)
    objects_table = out_dir / f"{output_stem}_objects.parquet"
    objects_json = out_dir / f"{output_stem}_objects_space.json"
    membership_table = out_dir / f"{output_stem}_object_membership.parquet"
    edges_table = (
        out_dir / f"{output_stem}_object_edges.parquet" if write_edge_table else None
    )

    processing_metadata = ProcessingProvenance(
        stage="deduplicate_across_planes",
        source_name=source_name,
        parameters=parameters,
        summary=summarize_deduplication_result(dataset.points, result),
    )
    objects_metadata = replace(
        dataset.metadata,
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="cleaned_objects",
        ),
        processing=processing_metadata,
    )

    result.objects.to_parquet(objects_table, index=False)
    objects_metadata.to_json(objects_json)
    result.membership.to_parquet(membership_table, index=False)
    if edges_table is not None:
        result.edges.to_parquet(edges_table, index=False)

    return DeduplicationOutputPaths(
        objects_table=objects_table,
        objects_json=objects_json,
        membership_table=membership_table,
        edges_table=edges_table,
    )


def save_colocalization_outputs(
    dataset_a: PointCloudDataset,
    dataset_b: PointCloudDataset,
    result: ColocalizationResult,
    out_dir: Path,
    *,
    channel_a_name: str,
    channel_b_name: str,
    max_xy_distance_um: float,
    object_relationships: ObjectRelationshipResult | None = None,
    source_a: str | None = None,
    source_b: str | None = None,
    output_stem: str | None = None,
    write_qc: bool = True,
    marker_radius_px: int = 2,
) -> ColocalizationOutputPaths:
    """Save detection matches, object relationships, provenance, and QC."""

    if output_stem is None:
        output_stem = make_subject_output_stem(dataset_a.subject_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    matches_table = out_dir / f"{output_stem}_detection_matches.parquet"
    relationships_table = (
        out_dir / f"{output_stem}_object_relationships.parquet"
        if object_relationships is not None
        else None
    )
    metadata_path = out_dir / f"{output_stem}_colocalization_metadata.json"
    qc_report = (
        out_dir / f"{output_stem}_colocalization_qc.xlsx" if write_qc else None
    )
    qc_images_dir = out_dir / "qc_images" if write_qc else None

    result.matches.to_parquet(matches_table, index=False)
    if relationships_table is not None and object_relationships is not None:
        object_relationships.relationships.to_parquet(relationships_table, index=False)

    relationship_summary: dict[str, int] = {}
    if object_relationships is not None and not object_relationships.relationships.empty:
        relationship_summary = {
            str(status): int(count)
            for status, count in object_relationships.relationships[
                "relationship_status"
            ].value_counts().items()
        }
    metadata = {
        "schema_name": "spatialsignal.colocalization",
        "schema_version": "0.1.0",
        "subject_name": dataset_a.subject_name,
        "space": dataset_a.space.to_dict(),
        "channels": {
            "a": {"name": channel_a_name, "source": source_a},
            "b": {"name": channel_b_name, "source": source_b},
        },
        "processing": {
            "stage": "match_colocalized_detections",
            "parameters": {
                "max_xy_distance_um": float(max_xy_distance_um),
                "z_rule": "exact_plane",
                "matching_rule": "maximum_cardinality_minimum_distance_one_to_one",
            },
            "summary": {
                "a_detections": int(len(dataset_a.points)),
                "b_detections": int(len(dataset_b.points)),
                "accepted_detection_matches": int(len(result.matches)),
                "ambiguous_detection_matches": int(
                    result.matches["is_ambiguous"].sum()
                ),
                "object_relationship_status_counts": relationship_summary,
            },
        },
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    if write_qc and qc_images_dir is not None and qc_report is not None:
        from spatialsignal.qc import (
            write_colocalization_images,
            write_colocalization_report,
        )

        image_paths = write_colocalization_images(
            dataset_a,
            dataset_b,
            result,
            qc_images_dir,
            output_stem=output_stem,
            marker_radius_px=marker_radius_px,
        )
        write_colocalization_report(
            result.plane_summary,
            image_paths,
            qc_report,
            channel_a_name=channel_a_name,
            channel_b_name=channel_b_name,
        )

    return ColocalizationOutputPaths(
        matches_table=matches_table,
        relationships_table=relationships_table,
        metadata_path=metadata_path,
        qc_report=qc_report,
        qc_images_dir=qc_images_dir,
    )


def save_instance_region_quantification_outputs(
    assigned_objects: pd.DataFrame,
    region_summary: pd.DataFrame,
    metadata: DatasetMetadata,
    out_dir: Path,
    *,
    source_name: str,
    parameters: dict[str, Any] | None = None,
    output_stem: str | None = None,
    annotation: LabelVolume | None = None,
    write_qc: bool = True,
) -> InstanceRegionQuantificationOutputPaths:
    """Save region-assigned objects, regional summaries, and visual QC."""

    if write_qc and annotation is None:
        raise ValueError("annotation is required when write_qc is True")

    subject_name = source_name.removesuffix("_objects.parquet").removesuffix(".parquet")
    if output_stem is None:
        output_stem = make_subject_output_stem(subject_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    assigned_objects_table = out_dir / f"{output_stem}_objects_with_regions.parquet"
    assigned_objects_json = out_dir / f"{output_stem}_objects_with_regions_space.json"
    region_summary_csv = out_dir / f"{output_stem}_region_summary.csv"
    qc_png = out_dir / f"{output_stem}_quantification_qc.png" if write_qc else None

    processing_metadata = ProcessingProvenance(
        stage="assign_objects_to_regions",
        source_name=source_name,
        parameters=parameters,
        summary={
            "n_objects": int(len(assigned_objects)),
            "n_assigned": int((assigned_objects["assignment_status"] == "assigned").sum()),
            "n_background": int((assigned_objects["assignment_status"] == "background").sum()),
            "n_out_of_bounds": int(
                (assigned_objects["assignment_status"] == "out_of_bounds").sum()
            ),
        },
    )
    assigned_metadata = replace(
        metadata,
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="objects_with_region_ids",
        ),
        processing=processing_metadata,
    )

    assigned_objects.to_parquet(assigned_objects_table, index=False)
    assigned_metadata.to_json(assigned_objects_json)
    region_summary.to_csv(region_summary_csv, index=False)
    if qc_png is not None:
        from spatialsignal.quantification import write_region_quantification_qc

        assigned_dataset = PointCloudDataset(
            subject_name=subject_name,
            points=assigned_objects,
            metadata=assigned_metadata,
        )
        write_region_quantification_qc(assigned_dataset, annotation, qc_png)

    return InstanceRegionQuantificationOutputPaths(
        assigned_objects_table=assigned_objects_table,
        assigned_objects_json=assigned_objects_json,
        region_summary_csv=region_summary_csv,
        qc_png=qc_png,
    )
