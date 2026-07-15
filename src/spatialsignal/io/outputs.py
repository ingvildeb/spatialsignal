"""Reusable output writers for spatialsignal datasets."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
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
class InstanceRegionQuantificationOutputPaths:
    """Paths written for instance region-assignment outputs."""

    assigned_objects_table: Path
    assigned_objects_json: Path
    region_summary_csv: Path


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


def save_instance_region_quantification_outputs(
    assigned_objects: pd.DataFrame,
    region_summary: pd.DataFrame,
    metadata: DatasetMetadata,
    out_dir: Path,
    *,
    source_name: str,
    parameters: dict[str, Any] | None = None,
    output_stem: str | None = None,
) -> InstanceRegionQuantificationOutputPaths:
    """Save region-assigned object tables and per-region summaries."""

    subject_name = source_name.removesuffix("_objects.parquet").removesuffix(".parquet")
    if output_stem is None:
        output_stem = make_subject_output_stem(subject_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    assigned_objects_table = out_dir / f"{output_stem}_objects_with_regions.parquet"
    assigned_objects_json = out_dir / f"{output_stem}_objects_with_regions_space.json"
    region_summary_csv = out_dir / f"{output_stem}_region_summary.csv"

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

    return InstanceRegionQuantificationOutputPaths(
        assigned_objects_table=assigned_objects_table,
        assigned_objects_json=assigned_objects_json,
        region_summary_csv=region_summary_csv,
    )
