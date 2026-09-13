# Roadmap

This document tracks development status and planned work for `spatialsignal`.
The package overview and supported public interface live in the repository
README.

## Settled Direction

- `atlasspace` owns registration, transforms, and registration output folders.
- `atlaslevels` owns atlas hierarchy, region metadata, and ID conversion.
- `spatialsignal` owns segmentation-derived spatial representations and
  downstream quantification.
- Subject-space outputs are the canonical quantitative lane.
- Reference-space maps are primarily visualization and comparison products.
- Jacobian-adjusted outputs require explicit validation before becoming a
  supported production feature.

## Completed

- Package rename and public-facing documentation structure
- Zero-based canonical point coordinates with explicit metadata sidecars
- Instance-mask centroid extraction with configurable filename filtering
- 2D detection morphology extraction
- Cross-plane deduplication and object-level morphology aggregation
- Vectorized object aggregation validated on 31.6 million detections
- Subject-space count-map voxelization
- Semantic signal-point extraction and fraction-map voxelization
- Parquet storage for large computational tables
- `atlasspace` registration-folder integration
- Native-mask-grid to registration-grid object remapping
- Subject-space object-to-region assignment
- Per-region counts, volumes, densities, area, and eccentricity summaries
- One-step atlas enrichment through `atlaslevels`
- Optional hierarchy-level instance summaries through curated `atlaslevels` bundles
- Legacy MATLAB validation helpers
- Project-level SING instance workflow using the reusable package APIs

## Next

### 1. Real-data validation and performance evaluation

- Run the revised Parquet and parallel extraction workflow on full NeuN and
  Iba1 datasets.
- Record stage runtimes and identify any remaining dominant bottlenecks.
- Compare object counts and regional summaries with previously validated data.
- Spot-check area and eccentricity distributions for biological plausibility.

### 2. Reference-space visualization

- Design and validate subject-to-reference transformation as a separate feature.
- Establish transform direction, coordinate conventions, output contracts, and
  count-conservation behavior before implementation.
- Add smoothing, boundary handling, and group displays only after the core
  transformation has been validated across age-specific template grids.

### 3. Subject-space semantic quantification

- Define canonical regional metrics for semantic signal.
- Quantify subject-space fraction or burden maps against warped annotations.
- Keep semantic outputs separate from instance counts and morphology.

## Later Research Lanes

### Jacobian-adjusted maps

Validate transform direction, Jacobian convention, and expected behavior in toy
examples with known expansion and compression before exposing these maps as a
standard output.

### Group-level outputs

Add cohort summaries, group-average maps, and hotspot-style analyses only after
the per-subject reference-space and semantic contracts are stable.
