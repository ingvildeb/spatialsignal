# Coordinate Systems

## Canonical Subject-Space Coordinates

The canonical point-cloud table uses the following columns:

- `detection_id`: dataset-wide unique identifier for one raw 2D detection
- `seg_num`: label ID from the 2D Cellpose mask image
- `x`: image horizontal coordinate
- `y`: image vertical coordinate
- `z`: sequential slice index assigned from the natural sort order of mask files

These coordinates describe image-space positions only. They do not encode anatomical orientation
inside the table itself. Instead, orientation and resolution are recorded in the metadata JSON
written alongside the CSV.

For the current subject-space pipeline:

- `x` corresponds to image columns
- `y` corresponds to image rows
- `z` corresponds to slice order in the stack

This means the canonical table is expressed directly in image-axis terms rather than raw
row/column naming.

## Canonical Indexing Convention

`spatialsignal` treats zero-based indexing as the canonical default for new outputs.

That means:

- `x = 0` is the first image column
- `y = 0` is the first image row
- `z = 0` is the first slice

## Slice Assignment

Slice numbering is defined by naturally sorted mask-file order:

1. Find all files matching the configured mask pattern.
2. Sort them with natural sorting.
3. Assign slices sequentially.

## Relationship to the Legacy MATLAB CSV

The relationship to the legacy MATLAB workflow of te Kim lab is documented separately in
`docs/legacy_matlab_relationship.md`.

The short version is:

- the MATLAB CSV is a compatibility and validation target
- it is not the canonical schema for new `spatialsignal` outputs
- `one_based` indexing should be chosen only when reproducing or checking against that workflow

## Point-Cloud Metadata

Each point-cloud CSV is accompanied by a nested metadata JSON that separates:

- `space`
  - spatial/grid definition such as `space_name`, `orientation`, `axis_labels`, `indexing`,
    `units`, `shape`, and `resolution_um`
- `representation`
  - dataset-level semantic fields such as `kind` and `representation_type`
- `processing`
  - optional provenance for derived outputs

For raw subject image space, the recommended axis labels are `x`, `y`, and `z`, and the
orientation should use a BrainGlobe-style code such as `las`.

In this package, the orientation string follows the BrainGlobe origin convention where each letter describes the anatomical side at voxel index `0` for that axis.

So for `las`:

- voxel `[0, 0, 0]` is left, anterior, superior
- increasing indices move:
  - `x`: left -> right
  - `y`: anterior -> posterior
  - `z`: superior -> inferior

## Representation Types

For the current workflows, the main representation types are:

- `point_centroids`: one row per centroid-like instance summary
- `signal_points`: one row per signal-support point from a binary semantic mask
- `count_map`: a voxelized count representation
- `fraction_map`: a voxelized semantic support fraction representation

## Cleaned Object Outputs

Cross-plane deduplication does not change the spatial frame of the data. Cleaned object outputs
remain in the same `x,y,z` image-axis convention and use the same point-cloud space definition as
raw detections.

What changes is the identity of each row:

- raw point cloud: one row = one 2D detection in one plane
- cleaned object point cloud: one row = one deduplicated object aggregated across one or more
  nearby planes

The cleaned objects JSON therefore keeps the same spatial metadata fields while adding a
`processing` section describing how the derived output was created.

## Subject-Space Quantification vs Reference-Space Maps

The current package strategy distinguishes clearly between subject-space quantitative outputs and
reference-space aligned maps.

### Subject-Space Quantification

Subject-space quantification is intended to be the canonical biological output.

The planned atlas-aware workflow is:

1. start from subject-space objects or voxel maps
2. consume an `atlasspace` registration output folder
3. use the warped annotation in subject space
4. summarize object counts, region volume, density, and object morphology by region

This keeps the biological measurements tied to the subject's own tissue geometry.

### Reference-Space Aligned Maps

Reference-space maps are useful for visualization and cross-subject comparison after anatomical
alignment.

However, an unmodulated aligned map should not automatically be interpreted as native cell density.
After nonlinear warping, local expansion or compression can change how many points fall into a
reference-space voxel even when native tissue density was unchanged.

For that reason:

- unmodulated aligned maps are useful descriptive outputs
- subject-space region summaries remain the canonical quantitative outputs
- Jacobian-modulated reference-space maps are a later specialized lane that requires explicit
  validation

## Voxel-Map Axis Order

Internally, voxel maps follow the same explicit spatial axis convention as point clouds and
metadata:

- voxel arrays are indexed as `data[x, y, z]`
- `space.shape` is interpreted in the same `x, y, z` order
- `axis_labels` and `orientation` therefore describe the in-memory voxel map directly

This is a deliberate package convention. It favors clear spatial reasoning and consistency with the
canonical point-cloud columns over image-stack or matrix-style indexing conventions.

Some external image-oriented tools and array libraries commonly treat 3D arrays as `z, y, x`
because they extend 2D row/column indexing by adding the slice axis first. `spatialsignal` does
not use that convention internally.

When voxel maps are exported to formats or libraries that expect image-style axis order, any
required permutation should happen explicitly at the I/O boundary rather than silently inside the
core spatial model.

## NIfTI Export Orientation

When writing NIfTI outputs, the package converts from the repo's BrainGlobe origin-based
orientation convention into NIfTI's affine-based RAS world convention explicitly.

This means:

- `orientation` in `SpaceDefinition` still means where voxel index `0` is
- the NIfTI affine then encodes the resulting directions of increasing voxel indices in an RAS+
  world

Those are related conventions, but they are not the same thing, so they should not be interpreted
interchangeably.
