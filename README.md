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

## Repository Layout

```text
src/        Reusable package code
scripts/    Thin command-line entry points
configs/    TOML configuration files
docs/       Workflow and coordinate-system documentation
tests/      Unit tests
```
