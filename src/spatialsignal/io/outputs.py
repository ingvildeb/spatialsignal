"""Reusable output writers for spatialsignal datasets."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
import re
from typing import Any

import numpy as np

from spatialsignal.models import PointCloudDataset, ProcessingProvenance, SpaceDefinition, VoxelMap
from spatialsignal.pointcloud.deduplicate import DeduplicationResult, summarize_deduplication_result
from spatialsignal.voxelization import build_nifti_ras_affine


@dataclass(frozen=True)
class VoxelMapOutputPaths:
    """Paths written for a saved voxel map."""

    array_path: Path
    metadata_path: Path
    nifti_path: Path
    nifti_written: bool


@dataclass(frozen=True)
class DeduplicationOutputPaths:
    """Paths written for saved deduplication outputs."""

    objects_csv: Path
    objects_json: Path
    membership_csv: Path
    edges_csv: Path | None


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return True


def save_voxel_map_outputs(
    voxel_map: VoxelMap,
    out_dir: Path,
    *,
    name_suffix: str,
    output_stem: str | None = None,
) -> VoxelMapOutputPaths:
    """Save a voxel map as NumPy, metadata JSON, and optional NIfTI."""

    if output_stem is None:
        output_stem = make_subject_output_stem(voxel_map.subject_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{output_stem}_{name_suffix}"
    array_path = base_path.with_suffix(".npy")
    nifti_path = out_dir / f"{output_stem}_{name_suffix}.nii.gz"
    metadata_path = out_dir / f"{output_stem}_{name_suffix}_space.json"

    np.save(array_path, voxel_map.data)
    voxel_map.metadata.to_json(metadata_path)
    nifti_written = write_nifti_voxel_map(voxel_map.data, voxel_map.space, nifti_path)

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

    if output_stem is None:
        output_stem = make_subject_output_stem(dataset.subject_name)
    if source_name is None:
        source_name = f"{output_stem}_pointcloud.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    objects_csv = out_dir / f"{output_stem}_objects.csv"
    objects_json = out_dir / f"{output_stem}_objects_space.json"
    membership_csv = out_dir / f"{output_stem}_object_membership.csv"
    edges_csv = out_dir / f"{output_stem}_object_edges.csv" if write_edge_table else None

    processing_metadata = ProcessingProvenance(
        stage="deduplicate_across_planes",
        source_name=source_name,
        parameters=parameters,
        summary=summarize_deduplication_result(dataset.points, result),
    )
    objects_metadata = replace(dataset.metadata, processing=processing_metadata)

    result.objects.to_csv(objects_csv, index=False)
    objects_metadata.to_json(objects_json)
    result.membership.to_csv(membership_csv, index=False)
    if edges_csv is not None:
        result.edges.to_csv(edges_csv, index=False)

    return DeduplicationOutputPaths(
        objects_csv=objects_csv,
        objects_json=objects_json,
        membership_csv=membership_csv,
        edges_csv=edges_csv,
    )
