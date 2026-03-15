# lsfm_cell_mapping

A spatial point-cloud and mapping toolkit for whole-brain segmented cells.

## Milestone 1

The first milestone focuses on converting Cellpose mask outputs into a validated
3D point-cloud CSV with explicit image-space coordinates:

- read `masks_*.tif*`
- assign sequential slice indices from naturally sorted file order
- extract one centroid per labeled object
- export a canonical CSV with `seg_num,row,col,slice`
- write centroid QC images
- validate output against the legacy MATLAB centroid CSV

Atlas transforms, voxelized density maps, and downstream statistics are planned
for later milestones.

## Installation

Install in editable mode from the repo root:

```text
pip install -e ".[dev]"
```

## Build Point Cloud

Edit [`configs/build_pointcloud/default.toml`](/c:/Users/SmartBrain_32C_TR/Documents/GitHub/lsfm_cell_mapping/configs/build_pointcloud/default.toml) to point to your mask directory and output location, then run:

```text
python scripts/build_pointcloud.py --config configs/build_pointcloud/default.toml
```

For Windows paths in TOML, prefer single-quoted strings:

```text
mask_dir = 'C:\path\to\masks'
out_dir = 'C:\path\to\output'
```

### Config Fields

The current build config uses four sections:

- `[input]`
  - `mask_dir`: directory containing `masks_*.tif*`
  - `pattern`: glob used to discover mask files
- `[output]`
  - `out_dir`: directory where outputs are written
  - `output_name`: filename for the canonical CSV
- `[processing]`
  - `slice_start`: starting value for sequential slice numbering
  - `one_based`: whether exported `row`/`col` coordinates are 1-based
  - `max_workers`: number of worker processes for slice-wise centroid extraction
  - `show_progress`: whether to print CLI progress updates
  - `progress_interval`: progress print frequency in slices
- `[qc]`
  - `write_centroid_images`: whether to write centroid QC images

### Outputs

The build script currently writes:

- `pointcloud.csv`
  - canonical centroid table with columns `seg_num,row,col,slice`
- `centroids_*.tif`
  - slice-level centroid QC images corresponding to each input mask image

## Validate Against MATLAB

To compare the result against a legacy MATLAB `centroids.csv`:

```text
python scripts/validate_against_matlab.py --python-csv path/to/pointcloud.csv --matlab-csv path/to/centroids.csv --relabel-matlab-slices
```

The validator:

- remaps the legacy MATLAB columns `seg_num,x,y,z` into canonical meaning
- can relabel MATLAB slice values into natural sequential order
- reports exact matches and any remaining mismatches

## Status

Milestone 1 is functionally complete:

- centroid extraction matches the legacy MATLAB workflow on real data
- the config-driven build path is validated
- unit tests cover the core extraction and validation logic

## Repository Layout

```text
src/        Reusable package code
scripts/    Thin command-line entry points
configs/    TOML configuration files
docs/       Workflow and coordinate-system documentation
tests/      Unit tests
```
