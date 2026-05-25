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

## Point-Cloud Space Metadata

Milestone 1 writes a sidecar JSON next to each point-cloud CSV. The JSON should
store metadata describing both the point cloud's spatial frame and what the
rows represent, such as:

- `space_name`
- `orientation`
- `representation_type`
- `axis_labels`
- `indexing`
- `units`
- `shape`
- `resolution_um`

For raw subject image space, the recommended axis labels are:

- `x`
- `y`
- `z`

and the orientation should use a BrainGlobe-style code such as `las` where
voxel `[0, 0, 0]` is left, anterior, and superior.

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
adds a `processing` section describing how the deduplication output was
derived.

## Future Coordinate Spaces

Later milestones will add explicit support for additional spaces such as:

- downsampled subject voxel space
- age-specific reference atlas spaces
- Allen CCFv3

Those spaces should be represented explicitly rather than inferred from file
names or directory structure.
