"""Example runner for voxelizing binary semantic mask stacks into a subject analysis fraction map."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.io import save_voxel_map_outputs
from spatialsignal.io.masks import find_mask_files
from spatialsignal.models import SpaceDefinition
from spatialsignal.utils import load_toml_config, require_config_value
from spatialsignal.voxelization import (
    make_subject_analysis_space,
    voxelize_signal_masks_to_fraction_map,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for signal-mask voxelization."""

    parser = argparse.ArgumentParser(
        description=(
            "Voxelize binary semantic segmentation masks directly into a derived "
            "subject analysis fraction map."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file.",
    )
    return parser.parse_args()


def main() -> int:
    """Voxelize a binary semantic mask stack into a fraction map."""

    args = parse_args()
    config = load_toml_config(Path(args.config))

    subject_name = require_config_value(config, "subject", "name")

    mask_dir = Path(require_config_value(config, "input", "mask_dir"))
    input_config = config.get("input", {})
    extensions = tuple(input_config.get("extensions", [".tif", ".tiff"]))
    prefix = input_config.get("prefix")
    suffix = input_config.get("suffix")

    out_dir = Path(require_config_value(config, "output", "out_dir"))

    source_space_name = require_config_value(config, "space", "name")
    orientation = require_config_value(config, "space", "orientation")
    resolution_um = require_config_value(config, "space", "resolution_um")
    indexing = config.get("space", {}).get("indexing", "zero_based")

    analysis_resolution_um = require_config_value(
        config,
        "analysis",
        "resolution_um",
    )
    analysis_space_name = config.get("analysis", {}).get(
        "space_name",
        "subject_analysis_space",
    )

    show_progress = bool(config.get("processing", {}).get("show_progress", True))
    progress_interval = int(config.get("processing", {}).get("progress_interval", 25))

    mask_files = find_mask_files(
        mask_dir,
        extensions=extensions,
        prefix=prefix,
        suffix=suffix,
    )
    source_space = SpaceDefinition.from_mask_files(
        space_name=source_space_name,
        orientation=orientation,
        resolution_um=resolution_um,
        indexing=indexing,
        mask_files=mask_files,
    )
    analysis_space = make_subject_analysis_space(
        source_space,
        analysis_resolution_um=analysis_resolution_um,
        space_name=analysis_space_name,
    )
    voxel_map = voxelize_signal_masks_to_fraction_map(
        mask_files,
        source_space,
        analysis_space,
        subject_name=subject_name,
        show_progress=show_progress,
        progress_interval=progress_interval,
    )

    output_paths = save_voxel_map_outputs(voxel_map, out_dir, name_suffix="fraction_map")

    summary = voxel_map.summary()
    processing_summary = voxel_map.metadata.processing.summary if voxel_map.metadata.processing else {}
    print("Signal-mask voxelization complete")
    print(f"  subject_name: {voxel_map.subject_name}")
    print(f"  source_space_name: {source_space.space_name}")
    print(f"  target_space_name: {voxel_map.space.space_name}")
    print(f"  source_shape_xyz: {tuple(source_space.shape)}")
    print(f"  target_shape_xyz: {tuple(voxel_map.space.shape)}")
    print(f"  source_resolution_um: {tuple(source_space.resolution_um)}")
    print(f"  target_resolution_um: {tuple(voxel_map.space.resolution_um)}")
    print(f"  input_signal_points: {processing_summary.get('input_signal_points', 0)}")
    print(f"  nonzero_voxels: {summary['nonzero_voxels']}")
    print(f"  max_fraction: {summary['max']}")
    print(f"  fraction_map_npy: {output_paths.array_path}")
    if output_paths.nifti_written:
        print(f"  fraction_map_nifti: {output_paths.nifti_path}")
    else:
        print("  fraction_map_nifti: skipped (install nibabel to enable NIfTI export)")
    print(f"  fraction_map_space_json: {output_paths.metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
