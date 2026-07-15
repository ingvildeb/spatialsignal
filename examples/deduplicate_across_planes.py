"""Example runner for cross-plane deduplication of raw point-cloud detections."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.io import save_deduplication_outputs
from spatialsignal.pointcloud import (
    PointCloudDataset,
    deduplicate_across_planes,
)
from spatialsignal.qc import select_qc_plane_pairs, write_pair_duplicate_qc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for cross-plane deduplication."""

    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate raw point-cloud detections across nearby z planes and write "
            "cleaned object tables plus provenance outputs."
        )
    )
    parser.add_argument(
        "--pointcloud-table",
        required=True,
        help="Path to a raw point-cloud Parquet table such as {subject}_pointcloud.parquet.",
    )
    parser.add_argument(
        "--space-json",
        required=True,
        help="Path to a point-cloud space JSON such as {subject}_pointcloud_space.json.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where deduplication outputs should be written.",
    )
    parser.add_argument(
        "--subject-name",
        default=None,
        help="Optional subject name override. If omitted, it is inferred from the table filename.",
    )
    parser.add_argument(
        "--max-plane-offset",
        type=int,
        required=True,
        help="Maximum forward plane offset to compare when building cross-plane links.",
    )
    parser.add_argument(
        "--max-xy-distance-um",
        type=float,
        required=True,
        help="Maximum lateral distance in microns allowed for a cross-plane link.",
    )
    parser.add_argument(
        "--max-n-planes",
        type=int,
        default=None,
        help="Optional biological cap on the number of unique planes allowed per cleaned object.",
    )
    parser.add_argument(
        "--write-edge-table",
        action="store_true",
        help="Write the accepted cross-plane edge table used to build cleaned objects.",
    )
    parser.add_argument(
        "--write-pair-qc",
        action="store_true",
        help="Write pairwise duplicate-vs-nonduplicate mask QC images for selected plane pairs.",
    )
    parser.add_argument(
        "--n-qc-pairs",
        type=int,
        default=3,
        help="Number of plane pairs to sample for pairwise QC when --write-pair-qc is enabled.",
    )
    parser.add_argument(
        "--qc-mask-dir",
        default=None,
        help="Directory containing the original TIFF masks used to build the raw point cloud.",
    )
    parser.add_argument(
        "--qc-output-dir",
        default=None,
        help="Optional base directory for QC outputs. If omitted, uses --out-dir.",
    )
    return parser.parse_args()


def main() -> int:
    """Run cross-plane deduplication and write outputs."""

    args = parse_args()
    if args.max_plane_offset < 1:
        raise ValueError(f"--max-plane-offset must be >= 1, got {args.max_plane_offset}")
    if args.max_xy_distance_um <= 0:
        raise ValueError(
            f"--max-xy-distance-um must be > 0, got {args.max_xy_distance_um}"
        )
    if args.max_n_planes is not None and args.max_n_planes < 1:
        raise ValueError(f"--max-n-planes must be >= 1, got {args.max_n_planes}")
    if args.n_qc_pairs < 1:
        raise ValueError(f"--n-qc-pairs must be >= 1, got {args.n_qc_pairs}")

    dataset = PointCloudDataset.from_files(
        table_path=Path(args.pointcloud_table),
        json_path=Path(args.space_json),
        subject_name=args.subject_name,
    )
    dataset.validate()

    result = deduplicate_across_planes(
        dataset.points,
        dataset.space,
        max_plane_offset=args.max_plane_offset,
        max_xy_distance_um=args.max_xy_distance_um,
        max_n_planes=args.max_n_planes,
    )

    out_dir = Path(args.out_dir)
    output_paths = save_deduplication_outputs(
        dataset,
        result,
        out_dir,
        source_name=Path(args.pointcloud_table).name,
        parameters={
            "max_plane_offset": args.max_plane_offset,
            "max_xy_distance_um": args.max_xy_distance_um,
            "max_n_planes": args.max_n_planes,
        },
        write_edge_table=args.write_edge_table,
    )

    qc_summary_path = None
    if args.write_pair_qc:
        if args.qc_mask_dir is None:
            raise ValueError("--qc-mask-dir is required when --write-pair-qc is enabled")
        qc_base_dir = Path(args.qc_output_dir) if args.qc_output_dir else out_dir
        qc_output_dir = qc_base_dir / "pair_qc"
        plane_pairs = select_qc_plane_pairs(dataset.points, n_pairs=args.n_qc_pairs)
        qc_summary_path = write_pair_duplicate_qc(
            points=dataset.points,
            membership=result.membership,
            mask_dir=Path(args.qc_mask_dir),
            plane_pairs=plane_pairs,
            output_dir=qc_output_dir,
        )

    print("Cross-plane deduplication complete")
    print(f"  subject_name: {dataset.subject_name}")
    print(f"  raw_detections: {len(dataset.points)}")
    print(f"  cleaned_objects: {len(result.objects)}")
    print(f"  accepted_edges: {len(result.edges)}")
    print(f"  max_n_planes: {args.max_n_planes}")
    print(f"  objects_table: {output_paths.objects_table}")
    print(f"  objects_space_json: {output_paths.objects_json}")
    if output_paths.edges_table is not None:
        print(f"  edges_table: {output_paths.edges_table}")
    print(f"  object_membership_table: {output_paths.membership_table}")
    if qc_summary_path is not None:
        print(f"  pair_qc_summary_json: {qc_summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
