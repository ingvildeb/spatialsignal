"""Example runner for voxelizing centroid point clouds into a subject analysis space."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.io import save_voxel_map_outputs
from spatialsignal.models import PointCloudDataset
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
        "--pointcloud-table",
        required=True,
        help="Path to a point-cloud table such as {subject}_pointcloud.parquet.",
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
        help="Optional subject name override. If omitted, it is inferred from the table filename.",
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

    dataset = PointCloudDataset.from_files(
        table_path=Path(args.pointcloud_table),
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


if __name__ == "__main__":
    raise SystemExit(main())
