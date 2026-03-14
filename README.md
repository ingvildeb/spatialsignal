# lsfm_cell_mapping

A spatial point-cloud and mapping toolkit for whole-brain segmented cells.

## Current Scope

This repository is being built in small, testable milestones.

The first milestone focuses on converting Cellpose mask outputs into a clean
3D point-cloud CSV with explicit image-space coordinates:

- read `masks_*.tif*`
- assign sequential slice indices from naturally sorted file order
- extract one centroid per labeled object
- export a canonical CSV with `seg_num,row,col,slice`
- optionally write centroid QC images
- validate output against the legacy MATLAB centroid CSV

Atlas transforms, voxelized density maps, and downstream statistics are planned
for later milestones.

## Current Usage

Install in editable mode from the repo root:

```text
pip install -e ".[dev]"
```

Edit [`configs/build_pointcloud/default.toml`](/c:/Users/SmartBrain_32C_TR/Documents/GitHub/lsfm_cell_mapping/configs/build_pointcloud/default.toml) to point to your mask directory and output location, then run:

```text
python scripts/build_pointcloud.py --config configs/build_pointcloud/default.toml
```

For Windows paths in TOML, prefer single-quoted strings:

```text
mask_dir = 'C:\path\to\masks'
out_dir = 'C:\path\to\output'
```

To compare the result against a legacy MATLAB `centroids.csv`:

```text
python scripts/validate_against_matlab.py --python-csv path/to/pointcloud.csv --matlab-csv path/to/centroids.csv --relabel-matlab-slices
```

## Repository Layout

```text
src/        Reusable package code
scripts/    Thin command-line entry points
configs/    TOML configuration files
docs/       Workflow and coordinate-system documentation
tests/      Unit tests
```
