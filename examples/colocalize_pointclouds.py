"""Example runner for same-plane colocalization of two centroid point clouds."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.io import save_colocalization_outputs
from spatialsignal.models import PointCloudDataset
from spatialsignal.pointcloud import match_colocalized_detections
from spatialsignal.utils import load_toml_config, require_config_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match two centroid point clouds one-to-one within exact z planes."
    )
    parser.add_argument("--config", required=True, help="Path to a TOML config file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_toml_config(Path(args.config))

    a_table = Path(require_config_value(config, "input_a", "table"))
    a_json = Path(require_config_value(config, "input_a", "space_json"))
    a_name = str(require_config_value(config, "input_a", "name"))
    b_table = Path(require_config_value(config, "input_b", "table"))
    b_json = Path(require_config_value(config, "input_b", "space_json"))
    b_name = str(require_config_value(config, "input_b", "name"))
    max_distance = float(
        require_config_value(config, "matching", "max_xy_distance_um")
    )
    out_dir = Path(require_config_value(config, "output", "out_dir"))
    qc_config = config.get("qc", {})

    dataset_a = PointCloudDataset.from_files(a_table, a_json)
    dataset_b = PointCloudDataset.from_files(b_table, b_json)
    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=max_distance,
    )
    paths = save_colocalization_outputs(
        dataset_a,
        dataset_b,
        result,
        out_dir,
        channel_a_name=a_name,
        channel_b_name=b_name,
        max_xy_distance_um=max_distance,
        source_a=str(a_table),
        source_b=str(b_table),
        write_qc=bool(qc_config.get("write", True)),
        marker_radius_px=int(qc_config.get("marker_radius_px", 2)),
    )

    print("Colocalization complete")
    print(f"  accepted_matches: {len(result.matches)}")
    print(f"  matches_table: {paths.matches_table}")
    print(f"  metadata: {paths.metadata_path}")
    if paths.qc_report is not None:
        print(f"  qc_report: {paths.qc_report}")
        print(f"  qc_images: {paths.qc_images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
