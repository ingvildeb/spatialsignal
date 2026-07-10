"""Example runner for building signal-support points from binary semantic mask stacks."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.pointcloud import build_signal_points_from_masks
from spatialsignal.utils import load_toml_config, require_config_value


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for signal-point building."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical point_id,x,y,z signal-point CSV from binary semantic masks."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file.",
    )
    return parser.parse_args()


def main() -> int:
    """Build a signal-point CSV from a directory of semantic mask images."""

    args = parse_args()
    config = load_toml_config(Path(args.config))

    subject_name = require_config_value(config, "subject", "name")

    mask_dir = Path(require_config_value(config, "input", "mask_dir"))
    pattern = config.get("input", {}).get("pattern", "*.tif*")

    out_dir = Path(require_config_value(config, "output", "out_dir"))

    space_name = require_config_value(config, "space", "name")
    orientation = require_config_value(config, "space", "orientation")
    resolution_um = require_config_value(config, "space", "resolution_um")
    indexing = config.get("space", {}).get("indexing", "zero_based")

    slice_start_value = config.get("processing", {}).get("slice_start")
    slice_start = None if slice_start_value is None else int(slice_start_value)
    max_workers = int(config.get("processing", {}).get("max_workers", 1))
    show_progress = bool(config.get("processing", {}).get("show_progress", True))
    progress_interval = int(config.get("processing", {}).get("progress_interval", 25))

    build_signal_points_from_masks(
        mask_dir=mask_dir,
        out_dir=out_dir,
        subject_name=subject_name,
        space_name=space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        pattern=pattern,
        indexing=indexing,
        slice_start=slice_start,
        max_workers=max_workers,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
