"""Example runner for loading and validating a point-cloud dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.pointcloud import PointCloudDataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset validation."""

    parser = argparse.ArgumentParser(
        description="Load a point-cloud Parquet table and matching space JSON, validate them, and print a summary."
    )
    parser.add_argument(
        "--pointcloud-table",
        required=True,
        help="Path to a point-cloud table such as {subject}_pointcloud.parquet.",
    )
    parser.add_argument(
        "--space-json",
        required=True,
        help="Path to a point-cloud space JSON such as {subject}_pointcloud_space.json.",
    )
    parser.add_argument(
        "--subject-name",
        default=None,
        help="Optional subject name override. If omitted, it is inferred from the table filename.",
    )
    return parser.parse_args()


def main() -> int:
    """Load, validate, and summarize a point-cloud dataset."""

    args = parse_args()
    dataset = PointCloudDataset.from_files(
        table_path=Path(args.pointcloud_table),
        json_path=Path(args.space_json),
        subject_name=args.subject_name,
    )
    dataset.validate()
    summary = dataset.summary()

    print("Point-cloud dataset validation passed")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
