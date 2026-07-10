# Roadmap

This document tracks the near-term development plan for `spatialsignal`.

It is intentionally more internal and planning-oriented than the public-facing README.

## Current baseline

The current package already supports:

- centroid point-cloud generation from labeled instance masks
- signal-point generation from binary semantic masks
- cross-plane deduplication of likely repeated detections
- subject-aligned voxelization of centroid point clouds into count maps
- subject-aligned voxelization of semantic masks into fraction maps
- metadata sidecars, QC outputs, and MATLAB validation helpers

## Key design decisions

The current agreed direction is:

- `atlasspace` owns registration, transforms, and registration output folders.
- `atlaslevels` owns atlas label hierarchy and region rollups.
- `spatialsignal` owns segmentation-derived spatial representations and downstream quantification.
- Canonical quantitative outputs should stay in subject space.
- Reference-space voxel maps should be treated mainly as visualization and comparison outputs.
- Jacobian-modulated reference-space maps are a later validated research lane, not a first-pass
  production output.

## Near-term priorities

### 1. Registration output integration

Add a small integration layer that can consume an `atlasspace` registration output folder and
expose the pieces needed downstream, especially:

- warped annotation in subject space
- warped brain mask in subject space
- transform files and summary metadata
- target-space metadata and provenance

### 2. Subject-space region quantification for instance data

Build a workflow that:

1. starts from a cleaned object table
2. samples the warped annotation in subject space
3. assigns one region ID per object
4. writes per-object tables plus per-region summaries

This is expected to be the canonical first-pass quantitative output for cell-body workflows.

### 3. Subject-space region quantification for semantic data

Build a corresponding workflow for dense semantic outputs that summarizes subject-space signal
against the warped annotation, likely from subject-aligned fraction maps or related derived maps.

### 4. Reference-space visualization outputs

Add explicit support for reference-space visualization products, especially aligned maps built from
transformed instance points. These should be labeled clearly as aligned reference-space maps rather
than native-density-preserving maps.

### 5. Jacobian-modulated map research lane

Before exposing any Jacobian-modulated outputs, validate:

- transform direction and Jacobian convention
- whether the intended output is content-preserving or density-preserving
- expected behavior in toy examples with known expansion/compression

### 6. Group-level outputs later

Only after the per-subject contract is stable should the package grow group-level summaries,
cohort aggregation helpers, or hotspot-style analyses.

## Proposed implementation order

1. Documentation and naming cleanup
2. Registration-folder integration
3. Subject-space point/object-to-region assignment
4. Subject-space per-region count, burden, volume, and density summaries
5. Subject-space semantic summaries
6. Reference-space visualization outputs
7. Jacobian prototype lane
8. Group-level outputs later