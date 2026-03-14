"""CLI for building a canonical point cloud from Cellpose mask stacks."""

from __future__ import annotations

import argparse
from pathlib import Path

from lsfm_cell_mapping.pointcloud import build_pointcloud_from_masks


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for point-cloud building."""

    parser = argparse.ArgumentParser(
        description="Build a canonical seg_num,row,col,slice point-cloud CSV from Cellpose masks."
    )
    parser.add_argument(
        "--mask-dir",
        required=True,
        help="Directory containing Cellpose mask images such as masks_*.tif",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the canonical point-cloud CSV will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="masks_*.tif*",
        help="Glob pattern used to identify mask images.",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        default=1,
        help="Starting value for sequential slice numbering.",
    )
    parser.add_argument(
        "--zero-based",
        action="store_true",
        help="Export row/col coordinates as 0-based instead of 1-based.",
    )
    parser.add_argument(
        "--output-name",
        default="pointcloud.csv",
        help="Filename for the exported canonical CSV.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of worker processes to use for slice-wise centroid extraction.",
    )
    return parser.parse_args()


def main() -> int:
    """Build a point-cloud CSV from a directory of Cellpose masks."""

    args = parse_args()
    build_pointcloud_from_masks(
        mask_dir=Path(args.mask_dir),
        out_dir=Path(args.out_dir),
        pattern=args.pattern,
        slice_start=args.slice_start,
        one_based=not args.zero_based,
        max_workers=args.max_workers,
        output_name=args.output_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
