# Workflow

This document describes the workflows currently supported by `spatialsignal` and the
boundaries between `spatialsignal` and the rest of the LSFM ecosystem.

## What spatialsignal does

`spatialsignal` currently supports four core workflow types:

1. Build subject-space centroid point clouds from labeled instance-segmentation masks.
2. Build subject-space dense signal-point tables from binary semantic masks.
3. Deduplicate repeated cell-body detections across adjacent planes.
4. Voxelize subject-space point clouds or semantic masks into subject-aligned analysis grids.

These workflows produce explicit spatial representations together with metadata sidecars that
make the coordinate system, indexing convention, representation type, and processing provenance
visible. The canonical starter configs use zero-based `x,y,z` indexing for subject-space outputs.

The canonical interface is the reusable Python code in `src/spatialsignal/`. The runnable
`examples/` in the repo are thin wrappers around those APIs and currently serve mainly as practical
examples and testing helpers.

## Supported workflows

### Instance-segmentation workflow

Use this workflow when each labeled object in a 2D mask corresponds to one cell body or other
instance-like detection.

Typical steps:

1. Build a raw detection point cloud with `examples/build_pointcloud.py`.
2. Optionally validate the result against the legacy MATLAB centroid CSV.
3. Deduplicate repeated detections across nearby z planes with `examples/deduplicate_across_planes.py`.
4. Optionally voxelize the cleaned object table into a subject-aligned count map with `examples/voxelize_pointcloud.py`.

Primary outputs:

- raw detection point cloud CSV + metadata JSON
- cleaned object table CSV + metadata JSON
- optional subject-aligned count map outputs

### Semantic-mask workflow

Use this workflow when the mask represents dense semantic support such as positive signal,
process masks, or other non-instance binary segmentation outputs.

Typical steps:

1. Build dense signal-support points with `examples/build_signal_points.py` when an explicit
   point representation is useful.
2. Or voxelize semantic masks directly into a subject-aligned fraction map with
   `examples/voxelize_signal_masks.py`.

Primary outputs:

- dense signal-point CSV + metadata JSON
- subject-aligned fraction map outputs

### Quality-control and validation workflow

The package also provides QC and validation helpers for:

- centroid image generation
- pairwise duplicate QC for deduplication
- MATLAB centroid CSV comparison
- point-cloud dataset validation against the declared metadata sidecar

## Spatial interpretation

The current package is subject-space-first.

- Subject-space point tables and subject-space region summaries are intended to be the canonical
  quantitative outputs.
- Subject-aligned voxel maps are useful for visualization, QC, and later subject-space
  quantification.
- Reference-space maps are expected to be useful mainly for visualization and cross-subject
  comparison, not as the primary biological truth.

A fuller discussion of subject-space quantification versus reference-space aligned maps lives in
`docs/coordinate_systems.md`.

## Near-term direction

The next major workflow extension is expected to be atlas-aware subject-space quantification:

1. load an `atlasspace` registration output folder
2. use the warped annotation in subject space
3. assign objects or voxelized signal to regions
4. summarize subject-space object counts, region volume, density, and morphology by region

The detailed plan for that work lives in `docs/roadmap.md`.
