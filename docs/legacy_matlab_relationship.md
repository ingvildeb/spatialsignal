# Legacy MATLAB Relationship

`spatialsignal` keeps a deliberate compatibility and validation lane for the legacy Kim lab MATLAB
cell-mapping workflow, but that legacy format does not define the canonical package schema.

## What the MATLAB Pipeline Writes

The MATLAB pipeline writes a `centroids.csv` file with columns `seg_num,x,y,z`.

For interpretation, there are three important details:

- the stored coordinates are effectively one-based
- the legacy `x` column contains image-row information
- the legacy `y` column contains image-column information

So even though the file headers say `x` and `y`, the spatial meaning is swapped relative to the
canonical `spatialsignal` image-axis convention.

## Canonical spatialsignal Representation

`spatialsignal` uses the following canonical subject-space interpretation:

- `x` = image column
- `y` = image row
- `z` = natural-sort slice index
- `indexing = "zero_based"` by default for new outputs

This is the convention that downstream metadata, validation, voxelization, and future
quantification features are meant to follow.

## How Compatibility Is Handled

When comparing against the legacy MATLAB CSV, `spatialsignal` remaps:

- legacy MATLAB `x` -> canonical `y`
- legacy MATLAB `y` -> canonical `x`
- legacy MATLAB `z` -> canonical `z`

If the legacy file uses sparse or absolute slice labels rather than consecutive numbering,
`relabel_slices_in_natural_order()` can relabel those z values to a consecutive stack order for
comparison.

## When to Use one_based in spatialsignal

Use `indexing = "one_based"` only when you intentionally need to reproduce or validate against the
legacy MATLAB convention, or when you must hand off outputs to a workflow that still expects that
style.

For new `spatialsignal`-native workflows, the recommendation is:

- keep point tables and metadata in the canonical zero-based convention
- use the MATLAB validation utilities only as a comparison layer
- avoid letting legacy CSV semantics dictate the public package defaults
