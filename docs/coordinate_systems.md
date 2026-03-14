# Coordinate Systems

## Canonical Milestone 1 Coordinates

The canonical point-cloud table for milestone 1 uses the following columns:

- `seg_num`: label ID from the 2D Cellpose mask image
- `row`: row index of the centroid in image coordinates
- `col`: column index of the centroid in image coordinates
- `slice`: sequential slice index assigned from the natural sort order of mask files

These coordinates describe image-space positions only. They do not yet encode
anatomical orientation, atlas alignment, or physical units.

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
image-coordinate meaning. The Python package should treat that format as a
legacy compatibility target, not as the canonical schema.

## Centroid Images

Centroid images are quality-control outputs only. They are derived from the
point table and should not be treated as the primary data source.

## Future Coordinate Spaces

Later milestones will add explicit support for additional spaces such as:

- native LSFM image space
- downsampled subject voxel space
- age-specific reference atlas spaces
- Allen CCFv3

Those spaces should be represented explicitly rather than inferred from file
names or directory structure.
