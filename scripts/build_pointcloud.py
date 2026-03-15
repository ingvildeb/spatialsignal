"""CLI for building a canonical point cloud from Cellpose mask stacks."""

from __future__ import annotations

import argparse
from pathlib import Path

from lsfm_cell_mapping.pointcloud import build_pointcloud_from_masks
from lsfm_cell_mapping.utils import load_toml_config, require_config_value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for point-cloud building."""

    parser = argparse.ArgumentParser(
        description="Build a canonical seg_num,row,col,slice point-cloud CSV from Cellpose masks."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file.",
    )
    return parser.parse_args()


def main() -> int:
    """Build a point-cloud CSV from a directory of Cellpose masks."""

    args = parse_args()
    config = load_toml_config(Path(args.config))

    subject_name = require_config_value(config, "subject", "name")

    mask_dir = Path(require_config_value(config, "input", "mask_dir"))
    pattern = config.get("input", {}).get("pattern", "masks_*.tif*")

    out_dir = Path(require_config_value(config, "output", "out_dir"))

    space_name = require_config_value(config, "space", "name")
    orientation = require_config_value(config, "space", "orientation")
    resolution_um = require_config_value(config, "space", "resolution_um")

    slice_start = int(config.get("processing", {}).get("slice_start", 1))
    one_based = bool(config.get("processing", {}).get("one_based", True))
    max_workers = int(config.get("processing", {}).get("max_workers", 1))
    show_progress = bool(config.get("processing", {}).get("show_progress", True))
    progress_interval = int(config.get("processing", {}).get("progress_interval", 25))
    write_qc_images = bool(config.get("qc", {}).get("write_centroid_images", False))

    build_pointcloud_from_masks(
        mask_dir=mask_dir,
        out_dir=out_dir,
        subject_name=subject_name,
        space_name=space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        pattern=pattern,
        slice_start=slice_start,
        one_based=one_based,
        max_workers=max_workers,
        write_qc_images=write_qc_images,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
