# Coordinate Systems

## Canonical Milestone 1 Coordinates

The canonical point-cloud table for milestone 1 uses the following columns:

- `detection_id`: dataset-wide unique identifier for one raw 2D detection
- `seg_num`: label ID from the 2D Cellpose mask image
- `x`: image horizontal coordinate
- `y`: image vertical coordinate
- `z`: sequential slice index assigned from the natural sort order of mask files

These coordinates describe image-space positions only. They do not yet encode
anatomical orientation directly in the table itself. Instead, those properties should
be recorded in a point-cloud space metadata JSON written alongside the CSV.

For the current subject-space pipeline:

- `x` corresponds to image columns
- `y` corresponds to image rows
- `z` corresponds to slice order in the stack

This means the canonical table is now expressed directly in image-axis terms
rather than raw row/column naming.

## Indexing Convention

Exported coordinates are intended to match the legacy MATLAB workflow and are
therefore treated as 1-based indices in milestone 1 outputs.

## Slice Assignment

Slice numbering is defined by naturally sorted mask-file order:

1. Find all files matching the configured mask pattern.
2. Sort them with natural sorting.
3. Assign slices sequentially from `slice_start`.

This intentionally avoids dependence on filename-derived z parsing rules.

## Relationship to the Legacy MATLAB CSV

The MATLAB pipeline writes a file named `centroids.csv` with columns
`seg_num,x,y,z`, but the stored values are actually:

- `seg_num`
- centroid row
- centroid column
- slice index

In other words, the legacy `x` and `y` headers do not match the underlying
image-coordinate meaning. During validation, the Python package remaps:

- legacy MATLAB `x` -> canonical `y`
- legacy MATLAB `y` -> canonical `x`
- legacy MATLAB `z` -> canonical `z`

The MATLAB format should therefore be treated as a legacy compatibility target,
not as the canonical schema. When loading legacy MATLAB CSVs into the current
schema, the Python package assigns synthetic sequential `detection_id` values,
but those IDs are not used for Python-vs-MATLAB coordinate matching.

## Centroid Images

Centroid images are quality-control outputs only. They are derived from the
point table and should not be treated as the primary data source.

## Point-Cloud Metadata

Milestone 1 writes a sidecar JSON next to each point-cloud CSV. The canonical
JSON schema is nested and separates:

- `space`
  - spatial/grid definition such as:
    - `space_name`
    - `orientation`
    - `axis_labels`
    - `indexing`
    - `units`
    - `shape`
    - `resolution_um`
- `representation`
  - dataset-level semantic fields such as:
    - `kind`
    - `representation_type`
- `processing`
  - optional provenance for derived outputs

For raw subject image space, the recommended axis labels are:

- `x`
- `y`
- `z`

and the orientation should use a BrainGlobe-style code such as `las` where
voxel `[0, 0, 0]` is left, anterior, and superior.

In this package, that orientation string follows the BrainGlobe origin
convention:

- each letter describes the anatomical side at voxel index `0` for that axis
- it does **not** describe the direction of increasing voxel indices

So for `las`:

- voxel `[0, 0, 0]` is left, anterior, superior
- increasing indices move:
  - `x`: left -> right
  - `y`: anterior -> posterior
  - `z`: superior -> inferior

For the current cell-body workflow, the recommended representation type is:

- `point_centroids`

This distinguishes centroid-based instance summaries from future coordinate
tables such as dense signal-support points (`signal_points`) or skeletonized
ramification coordinates (`skeleton_points`).

## Cleaned Object Outputs

Cross-plane deduplication does not change the spatial frame of the data. The
cleaned object outputs remain in the same `x,y,z` image-axis convention and use
the same point-cloud space definition as the raw detections.

What changes is the identity of each row:

- raw point cloud: one row = one 2D detection in one plane
- cleaned object point cloud: one row = one deduplicated object aggregated
  across one or more nearby planes

The cleaned objects JSON therefore keeps the same spatial metadata fields and
keeps the same `space` section while adding a `processing` section describing how the deduplication output was
derived.

## Future Coordinate Spaces

Later milestones will add explicit support for additional spaces such as:

- downsampled subject voxel space
- age-specific reference atlas spaces
- Allen CCFv3

Those spaces should be represented explicitly rather than inferred from file
names or directory structure.

## Voxel Map Axis Order

Internally, voxel maps follow the same explicit spatial axis convention as
point clouds and metadata:

- voxel arrays are indexed as `data[x, y, z]`
- `space.shape` is interpreted in the same `x, y, z` order
- `axis_labels` and `orientation` therefore describe the in-memory voxel map
  directly

This is a deliberate package convention. It favors clear spatial reasoning and
consistency with the canonical point-cloud columns over image-stack or
matrix-style indexing conventions.

Some external image-oriented tools and array libraries commonly treat 3D arrays
as `z, y, x` because they extend 2D row/column indexing (`y, x`) by adding the
slice axis first. `lsfm_cell_mapping` does **not** use that convention
internally.

When voxel maps are later exported to formats or libraries that expect
image-style axis order, any required permutation should happen explicitly at the
I/O boundary rather than silently inside the core spatial model.

## NIfTI Export Orientation

When writing NIfTI outputs, the package converts from the repo's BrainGlobe
origin-based orientation convention into NIfTI's affine-based RAS world
convention explicitly.

This means:

- `orientation` in `SpaceDefinition` still means "where voxel index 0 is"
- the NIfTI affine then encodes the resulting directions of increasing voxel
  indices in an RAS+ world

Those are related conventions, but they are not the same thing, so they should
not be interpreted interchangeably.
