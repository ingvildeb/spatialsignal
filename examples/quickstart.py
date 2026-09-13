"""Run a complete spatialsignal workflow on a tiny synthetic mask stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from spatialsignal.io import save_deduplication_outputs, save_voxel_map_outputs
from spatialsignal.models import PointCloudDataset
from spatialsignal.pointcloud import (
    build_pointcloud_from_masks,
    deduplicate_across_planes,
)
from spatialsignal.voxelization import make_subject_analysis_space, voxelize_to_space


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run spatialsignal on a generated four-plane instance-mask stack."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("quickstart-output"),
        help="New directory for generated masks and outputs.",
    )
    return parser.parse_args()


def write_synthetic_masks(mask_dir: Path) -> None:
    """Write four small masks containing two objects seen on adjacent planes."""

    mask_dir.mkdir(parents=True)
    masks = [np.zeros((32, 32), dtype=np.uint16) for _ in range(4)]

    masks[0][5:8, 6:9] = 1
    masks[1][6:9, 6:9] = 7
    masks[2][20:24, 18:22] = 3
    masks[3][20:24, 19:23] = 11

    for index, mask in enumerate(masks):
        tifffile.imwrite(mask_dir / f"mask_{index:03d}.tif", mask)


def main() -> int:
    """Generate masks, extract detections, deduplicate, and write a count map."""

    args = parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {out_dir}. Choose a new --out-dir."
        )

    mask_dir = out_dir / "masks"
    result_dir = out_dir / "results"
    write_synthetic_masks(mask_dir)

    subject_name = "example_subject"
    build_pointcloud_from_masks(
        mask_dir=mask_dir,
        out_dir=result_dir,
        subject_name=subject_name,
        space_name="example_native",
        orientation="las",
        resolution_um=[1.0, 1.0, 2.0],
        max_workers=1,
    )

    detections = PointCloudDataset.from_files(
        table_path=result_dir / f"{subject_name}_pointcloud.parquet",
        json_path=result_dir / f"{subject_name}_pointcloud_space.json",
    )
    deduplication = deduplicate_across_planes(
        detections.points,
        detections.space,
        max_plane_offset=1,
        max_xy_distance_um=2.0,
        max_n_planes=2,
    )
    dedup_paths = save_deduplication_outputs(
        detections,
        deduplication,
        result_dir,
        parameters={
            "max_plane_offset": 1,
            "max_xy_distance_um": 2.0,
            "max_n_planes": 2,
        },
    )

    objects = PointCloudDataset.from_files(
        table_path=dedup_paths.objects_table,
        json_path=dedup_paths.objects_json,
        subject_name=subject_name,
    )
    analysis_space = make_subject_analysis_space(
        objects.space,
        analysis_resolution_um=[4.0, 4.0, 4.0],
        space_name="example_analysis",
    )
    count_map = voxelize_to_space(objects, analysis_space)
    count_paths = save_voxel_map_outputs(
        count_map,
        result_dir,
        name_suffix="count_map",
        formats=("npy", "nifti"),
    )

    print(f"Raw detections: {len(detections.points)}")
    print(f"Cleaned objects: {len(objects.points)}")
    print(f"Count-map total: {int(count_map.data.sum())}")
    print(f"Results: {result_dir}")
    print(f"Metadata: {count_paths.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
