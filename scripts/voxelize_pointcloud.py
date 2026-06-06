"""CLI for voxelizing centroid point clouds into a subject analysis space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lsfm_cell_mapping.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
)
from lsfm_cell_mapping.pointcloud.build import make_subject_output_stem
from lsfm_cell_mapping.voxelization import (
    build_nifti_ras_affine,
    make_subject_analysis_space,
    voxelize_to_space,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for centroid voxelization."""

    parser = argparse.ArgumentParser(
        description=(
            "Voxelize a centroid point cloud into a derived subject analysis space "
            "and write a count-map NumPy array plus metadata sidecar."
        )
    )
    parser.add_argument(
        "--pointcloud-csv",
        required=True,
        help="Path to a point-cloud CSV such as {subject}_pointcloud.csv.",
    )
    parser.add_argument(
        "--space-json",
        required=True,
        help="Path to a point-cloud metadata JSON such as {subject}_pointcloud_space.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where voxelization outputs should be written.",
    )
    parser.add_argument(
        "--analysis-resolution-um",
        nargs=3,
        type=float,
        required=True,
        metavar=("X_RES", "Y_RES", "Z_RES"),
        help="Target subject analysis resolution in microns, for example 20 20 20.",
    )
    parser.add_argument(
        "--subject-name",
        default=None,
        help="Optional subject name override. If omitted, it is inferred from the CSV filename when possible.",
    )
    parser.add_argument(
        "--analysis-space-name",
        default="subject_analysis_space",
        help="Name to assign to the derived subject analysis space.",
    )
    return parser.parse_args()


def main() -> int:
    """Voxelize a centroid point cloud into a count map."""

    args = parse_args()

    dataset = load_pointcloud_dataset(
        csv_path=Path(args.pointcloud_csv),
        json_path=Path(args.space_json),
        subject_name=args.subject_name,
    )
    dataset.validate_spatial_points()

    analysis_space = make_subject_analysis_space(
        dataset.space,
        analysis_resolution_um=[float(value) for value in args.analysis_resolution_um],
        space_name=args.analysis_space_name,
    )
    voxel_map = voxelize_to_space(dataset, analysis_space)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = make_subject_output_stem(dataset.subject_name)
    map_path = out_dir / f"{output_stem}_count_map.npy"
    nifti_path = out_dir / f"{output_stem}_count_map.nii.gz"
    metadata_path = out_dir / f"{output_stem}_count_map_space.json"

    np.save(map_path, voxel_map.data)
    nifti_written = write_nifti_count_map(voxel_map.data, voxel_map.space, nifti_path)
    voxel_map.metadata.to_json(metadata_path)

    summary = voxel_map.summary()
    print("Point-cloud voxelization complete")
    print(f"  subject_name: {voxel_map.subject_name}")
    print(f"  source_space_name: {dataset.space.space_name}")
    print(f"  target_space_name: {voxel_map.space.space_name}")
    print(f"  target_shape_xyz: {tuple(voxel_map.space.shape)}")
    print(f"  target_resolution_um: {tuple(voxel_map.space.resolution_um)}")
    print(f"  total_count: {int(voxel_map.data.sum())}")
    print(f"  nonzero_voxels: {summary['nonzero_voxels']}")
    print(f"  count_map_npy: {map_path}")
    if nifti_written:
        print(f"  count_map_nifti: {nifti_path}")
    else:
        print("  count_map_nifti: skipped (install nibabel to enable NIfTI export)")
    print(f"  count_map_space_json: {metadata_path}")

    return 0


def load_pointcloud_dataset(
    csv_path: Path,
    json_path: Path,
    *,
    subject_name: str | None = None,
) -> PointCloudDataset:
    """Load a point-cloud dataset, upgrading older flat sidecars when needed."""

    try:
        return PointCloudDataset.from_files(
            csv_path=csv_path,
            json_path=json_path,
            subject_name=subject_name,
        )
    except KeyError as exc:
        if str(exc) != "'space'":
            raise

    points = pd.read_csv(csv_path)
    metadata = load_legacy_flat_metadata(json_path)
    if subject_name is None:
        subject_name = infer_subject_name_from_pointcloud_path(csv_path)

    return PointCloudDataset(
        subject_name=subject_name,
        points=points,
        metadata=metadata,
    )


def load_legacy_flat_metadata(json_path: Path) -> DatasetMetadata:
    """Load an older flat metadata sidecar into the current metadata model."""

    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    processing_data = data.get("processing")
    processing = None
    if processing_data is not None:
        processing = ProcessingProvenance(
            stage=processing_data["stage"],
            source_name=processing_data.get("source_name")
            or processing_data.get("source_pointcloud_csv"),
            parameters=processing_data.get("parameters"),
            summary=processing_data.get("summary"),
        )

    return DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version=data.get("schema_version", "0.1.0"),
        space=SpaceDefinition(
            space_name=data["space_name"],
            orientation=data["orientation"],
            axis_labels=data["axis_labels"],
            indexing=data["indexing"],
            units=data["units"],
            shape=data["shape"],
            resolution_um=data["resolution_um"],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type=data.get("representation_type", "point_centroids"),
        ),
        processing=processing,
    )


def infer_subject_name_from_pointcloud_path(csv_path: Path) -> str:
    """Infer subject name from common point-cloud and cleaned-object CSV stems."""

    suffixes = ("_pointcloud", "_objects")
    stem = csv_path.stem
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def write_nifti_count_map(
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


if __name__ == "__main__":
    raise SystemExit(main())
