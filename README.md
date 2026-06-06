# lsfm_cell_mapping

A spatial point-cloud and mapping toolkit for whole-brain segmented cells.

## Milestone 1

The first milestone focuses on converting Cellpose mask outputs into a validated
3D point-cloud representation in subject image space:

- read `masks_*.tif*`
- assign sequential slice indices from naturally sorted file order
- extract one centroid per labeled object
- export a canonical CSV with required columns `detection_id,seg_num,x,y,z`
- include standard object-property columns derived from the 2D masks
- export a space metadata JSON alongside the CSV
- write centroid QC images
- validate output against the legacy MATLAB centroid CSV

Atlas transforms, voxelized density maps, and downstream statistics are planned
for later milestones.

## Deduplicate Across Planes

After building a raw point cloud, a second stage can merge likely duplicate
cell-body detections across nearby z planes and write a cleaned object table.

Example:

```text
python scripts/deduplicate_across_planes.py ^
  --pointcloud-csv path/to/{subject_name}_pointcloud.csv ^
  --space-json path/to/{subject_name}_pointcloud_space.json ^
  --out-dir path/to/output ^
  --max-plane-offset 1 ^
  --max-xy-distance-um 3.0 ^
  --max-n-planes 2 ^
  --write-pair-qc ^
  --qc-mask-dir path/to/masks
```

This stage is intended for cross-plane deduplication only. It does not try to
split touching cells within a single plane.

## Installation

Install in editable mode from the repo root:

```text
pip install -e .
```

## Build Point Cloud

Edit [`configs/build_pointcloud/default.toml`](/c:/Users/SmartBrain_32C_TR/Documents/GitHub/lsfm_cell_mapping/configs/build_pointcloud/default.toml) to point to your mask directory and output location, then run:

```text
python scripts/build_pointcloud.py --config configs/build_pointcloud/default.toml
```

For Windows paths in TOML, use single-quoted strings:

```text
mask_dir = 'C:\path\to\masks'
out_dir = 'C:\path\to\output'
```

### Config Fields

The current build config uses five sections:

- `[subject]`
  - `name`: subject identifier used to generate standardized output filenames

- `[input]`
  - `mask_dir`: directory containing `masks_*.tif*`
  - `pattern`: glob used to discover mask files
- `[output]`
  - `out_dir`: directory where outputs are written
- `[space]`
  - `name`: space label for the current point cloud, e.g. `subject_space`
  - `orientation`: BrainGlobe-style orientation code, e.g. `las`
  - `resolution_um`: voxel spacing in axis order `[x, y, z]`
- `[data]`
  - `representation_type`: what one row of coordinates represents, e.g. `point_centroids`
- `[processing]`
  - `slice_start`: starting value for sequential slice numbering
  - `one_based`: whether exported `x`/`y`/`z` coordinates are 1-based
  - `max_workers`: number of worker processes for slice-wise centroid extraction
  - `show_progress`: whether to print CLI progress updates
  - `progress_interval`: progress print frequency in slices
- `[qc]`
  - `write_centroid_images`: whether to write centroid QC images

### Outputs

The build script currently writes:

- `{subject_name}_pointcloud.csv`
  - canonical point-cloud table with required columns `detection_id,seg_num,x,y,z`
  - standard additional columns currently include:
    - `area_px`
    - `x_float`
    - `y_float`
    - `major_axis_length_px`
    - `minor_axis_length_px`
    - `eccentricity`
- `{subject_name}_pointcloud_space.json`
  - metadata describing the point cloud's spatial frame and coordinate representation
- `centroids_*.tif`
  - slice-level centroid QC images corresponding to each input mask image

## Deduplication Outputs

The deduplication script writes:

- `{subject_name}_objects.csv`
  - cleaned object table after cross-plane deduplication
- `{subject_name}_objects_space.json`
  - spatial metadata plus a `processing` block recording:
    - deduplication stage name
    - source raw point-cloud filename
    - parameters used
    - summary counts for raw detections, cleaned objects, and accepted edges
- `{subject_name}_object_membership.csv`
  - detection-to-object membership mapping showing which raw `detection_id`
    values ended up in which cleaned `object_id`

Optional debug output:

- `{subject_name}_object_edges.csv`
  - accepted cross-plane links between raw detections
  - written only when `--write-edge-table` is enabled

Optional debug/provenance flags:

- `--write-edge-table`

Optional pairwise duplicate QC can also be written with:

- `--write-pair-qc`
- `--qc-mask-dir`
- optional `--qc-output-dir`

QC images are written into a `pair_qc/` subfolder. The current mask-based
pairwise QC is most directly interpretable for workflows that biologically cap
objects to at most two contributing planes.

## Validate Against MATLAB

To compare the result against a legacy MATLAB `centroids.csv`:

```text
python scripts/validate_against_matlab.py --python-csv path/to/{subject_name}_pointcloud.csv --matlab-csv path/to/centroids.csv --relabel-matlab-slices
```

The validator:

- remaps the legacy MATLAB columns `seg_num,x,y,z` into canonical spatial meaning
  while assigning synthetic `detection_id` values for schema compatibility
- can relabel MATLAB slice values into natural sequential order
- reports exact matches and any remaining mismatches

## Status

Milestone 1 is functionally complete:

- centroid extraction matches the legacy MATLAB workflow on real data
- the config-driven build path is validated
- the point-cloud space JSON is written alongside the CSV
- unit tests cover the core extraction and validation logic

## Repository Layout

```text
src/        Reusable package code
scripts/    Thin command-line entry points
configs/    TOML configuration files
docs/       Workflow and coordinate-system documentation
tests/      Unit tests
```
