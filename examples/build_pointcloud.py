"""Example runner for building a canonical point cloud from Cellpose mask stacks."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.pointcloud import build_pointcloud_from_masks
from spatialsignal.utils import load_toml_config, require_config_value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for point-cloud building."""

    parser = argparse.ArgumentParser(
        description="Build a canonical detection_id,seg_num,x,y,z point-cloud CSV from Cellpose masks."
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
    input_config = config.get("input", {})
    extensions = tuple(input_config.get("extensions", [".tif", ".tiff"]))
    prefix = input_config.get("prefix")
    suffix = input_config.get("suffix")

    out_dir = Path(require_config_value(config, "output", "out_dir"))

    space_name = require_config_value(config, "space", "name")
    orientation = require_config_value(config, "space", "orientation")
    resolution_um = require_config_value(config, "space", "resolution_um")
    indexing = config.get("space", {}).get("indexing", "zero_based")
    representation_type = config.get("data", {}).get("representation_type", "point_centroids")

    slice_start_value = config.get("processing", {}).get("slice_start")
    slice_start = None if slice_start_value is None else int(slice_start_value)
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
        representation_type=representation_type,
        extensions=extensions,
        prefix=prefix,
        suffix=suffix,
        indexing=indexing,
        slice_start=slice_start,
        max_workers=max_workers,
        write_qc_images=write_qc_images,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
