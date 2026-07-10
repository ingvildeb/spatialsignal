"""Example runner for voxelizing centroid point clouds into a subject analysis space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spatialsignal.io import save_voxel_map_outputs
from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
)
from spatialsignal.voxelization import (
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
    output_paths = save_voxel_map_outputs(voxel_map, out_dir, name_suffix="count_map")

    summary = voxel_map.summary()
    print("Point-cloud voxelization complete")
    print(f"  subject_name: {voxel_map.subject_name}")
    print(f"  source_space_name: {dataset.space.space_name}")
    print(f"  target_space_name: {voxel_map.space.space_name}")
    print(f"  target_shape_xyz: {tuple(voxel_map.space.shape)}")
    print(f"  target_resolution_um: {tuple(voxel_map.space.resolution_um)}")
    print(f"  total_count: {int(voxel_map.data.sum())}")
    print(f"  nonzero_voxels: {summary['nonzero_voxels']}")
    print(f"  count_map_npy: {output_paths.array_path}")
    if output_paths.nifti_written:
        print(f"  count_map_nifti: {output_paths.nifti_path}")
    else:
        print("  count_map_nifti: skipped (install nibabel to enable NIfTI export)")
    print(f"  count_map_space_json: {output_paths.metadata_path}")

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
        schema_name="spatialsignal.dataset_metadata",
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
if __name__ == "__main__":
    raise SystemExit(main())
