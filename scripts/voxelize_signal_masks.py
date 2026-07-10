"""CLI for voxelizing binary semantic mask stacks into a subject analysis fraction map."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lsfm_cell_mapping.io.masks import find_mask_files
from lsfm_cell_mapping.models import SpaceDefinition
from lsfm_cell_mapping.pointcloud.build import make_subject_output_stem
from lsfm_cell_mapping.utils import load_toml_config, require_config_value
from lsfm_cell_mapping.voxelization import (
    build_nifti_ras_affine,
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
    pattern = config.get("input", {}).get("pattern", "*.tif*")

    out_dir = Path(require_config_value(config, "output", "out_dir"))

    source_space_name = require_config_value(config, "space", "name")
    orientation = require_config_value(config, "space", "orientation")
    resolution_um = require_config_value(config, "space", "resolution_um")
    indexing = config.get("space", {}).get("indexing", "one_based")

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

    mask_files = find_mask_files(mask_dir, pattern=pattern)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = make_subject_output_stem(subject_name)
    map_path = out_dir / f"{output_stem}_fraction_map.npy"
    nifti_path = out_dir / f"{output_stem}_fraction_map.nii.gz"
    metadata_path = out_dir / f"{output_stem}_fraction_map_space.json"

    np.save(map_path, voxel_map.data)
    nifti_written = write_nifti_voxel_map(voxel_map.data, voxel_map.space, nifti_path)
    voxel_map.metadata.to_json(metadata_path)

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
    print(f"  fraction_map_npy: {map_path}")
    if nifti_written:
        print(f"  fraction_map_nifti: {nifti_path}")
    else:
        print("  fraction_map_nifti: skipped (install nibabel to enable NIfTI export)")
    print(f"  fraction_map_space_json: {metadata_path}")

    return 0


def write_nifti_voxel_map(
    data_xyz: np.ndarray,
    space: SpaceDefinition,
    output_path: Path,
) -> bool:
    """Write a voxel map as NIfTI if nibabel is available."""

    try:
        import nibabel as nib
    except ModuleNotFoundError:
        return False

    affine = build_nifti_ras_affine(space)
    image = nib.Nifti1Image(data_xyz, affine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return True


if __name__ == "__main__":
    raise SystemExit(main())
