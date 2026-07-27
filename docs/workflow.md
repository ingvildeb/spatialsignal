# Workflows

This document describes the workflows currently supported by `spatialsignal`
and the boundaries between `spatialsignal` and neighboring LSFM packages.

## Package Boundary

- `atlasspace` owns registration, transforms, and registration output folders.
- `atlaslevels` owns atlas hierarchy, region metadata, and ID conversion.
- `spatialsignal` owns segmentation-derived point tables, voxel maps, and
  subject-space quantification.

The canonical interface is the reusable Python code under `src/spatialsignal/`.
Scripts under `examples/` are thin API demonstrations and testing helpers.

## Instance-Segmentation Workflow

Use this workflow when each labeled object in a 2D mask represents a cell body
or another instance-like detection.

### 1. Build the raw point cloud

`build_pointcloud_from_masks()` naturally sorts the mask files, extracts one
centroid per nonzero label, and writes:

- `<subject>_pointcloud.parquet`
- `<subject>_pointcloud_space.json`

The Parquet table contains coordinates, area, major and minor axis lengths,
and eccentricity for each 2D detection. The JSON sidecar records the spatial
grid, indexing convention, representation type, and provenance.

### 2. Deduplicate detections across planes

`deduplicate_across_planes()` links nearby detections in adjacent planes and
aggregates them into biological objects. `save_deduplication_outputs()` writes:

- `<subject>_objects.parquet`
- `<subject>_objects_space.json`
- `<subject>_object_membership.parquet`
- optional `<subject>_object_edges.parquet`

The object table preserves native subject-space coordinates. Morphology is
averaged across the object's contributing 2D detections; these measurements
describe in-plane segmentation morphology, not reconstructed 3D cell shape.

### Colocalize two centroid channels

`match_colocalized_detections()` compares two compatible raw
`PointCloudDataset` instances only within equal z indices. It builds candidate
pairs from physical XY distance and selects a maximum-cardinality,
minimum-distance one-to-one matching independently in each plane. Z resolution
does not enter the colocalization distance.

`save_colocalization_outputs()` writes the accepted detection relationships as
Parquet, a dedicated JSON provenance sidecar, one coordinate-aligned transparent
PNG for every declared plane, and a filterable per-plane XLSX report. These
relational outputs avoid duplicating complete A-only, B-only, and union point
clouds.

After each channel is deduplicated, `map_detection_matches_to_objects()` uses
the channel membership tables to lift raw matching evidence to cleaned-object
relationships. Ordinary relationships are one-to-one. One-to-many or
many-to-many relationships are retained and explicitly classified for QC; the
package does not silently discard or force-resolve them.

### 3. Quantify objects by region

The registration integration layer consumes an `atlasspace` output folder and
loads its subject-space warped annotation. A separate brain mask is not needed
for region quantification: annotation background is excluded from the report by
default, while out-of-bounds objects remain available as a QC row.

`quantify_objects_by_region()` then:

1. remaps cleaned-object coordinates from the native mask grid into the
   annotation grid
2. assigns one region ID to each object
3. summarizes counts, region volume, density, median object area, and median
   object eccentricity
4. uses `atlaslevels` to add canonical Allen IDs, acronyms, names, and colors
5. optionally writes the compact region report as CSV together with its visual
   QC PNG

The returned `RegionQuantificationResult` retains the remapped object dataset,
the per-object assignments, the named region summary, and the optional CSV and
QC paths. Project workflows can additionally save the assigned-object table,
metadata sidecar, summary CSV, and automatic QC PNG with
`save_instance_region_quantification_outputs()`.

The primary outputs are:

- `<subject>_objects_with_regions.parquet`
- `<subject>_objects_with_regions_space.json`
- `<subject>_region_summary.csv`
- `<subject>_quantification_qc.png`

Subject-space regional summaries are the canonical biological quantitative
outputs.

### Optional hierarchy-level summaries

`summarize_objects_by_hierarchy_level()` produces an additional report at one
curated `atlaslevels` hierarchy level. The caller explicitly selects the
hierarchy preset and level for each invocation. The function maps source IDs
to canonical Allen hierarchy parents, aggregates annotation volumes and
object counts, recomputes density, and calculates morphology medians directly
from all underlying objects assigned to each parent.

Observed parent-region labels are preserved as explicit residual rows when a
selected hierarchy level splits that parent's descendants more finely. The
annotation is authoritative here, so an observed ancestor is retained even if
the packaged atlas metadata does not predict direct voxel support for it. This
ensures that hierarchy reports retain every labeled annotation voxel.

Hierarchy reports mark these residual-only rows with
`is_parent_residual = True`. The report's `region_id` is already the canonical
Allen ID after hierarchy mapping, so it does not duplicate that value in an
`allen_region_id` column. The selected level and source ID namespace remain
invocation settings rather than repeated columns in every CSV row.

The detailed report remains the canonical base result. Child-region densities
are never averaged, and child-region medians are never combined to approximate
parent morphology.

### 4. Create a subject-space count map

`voxelize_to_space()` aggregates deduplicated object centroids onto a declared
subject analysis grid. Count maps are the canonical stored subject-level voxel
representation. Density maps can be derived from counts and voxel volume when
needed, but do not need to be stored systematically.

## Semantic-Mask Workflow

Use this workflow when a binary mask represents dense signal support, such as
positive area or processes, rather than separate biological objects.

The package currently supports:

- `build_signal_points_from_masks()` for explicit signal-support Parquet tables
- direct voxelization of semantic masks into subject-space fraction maps
- metadata sidecars for both representations

Atlas-aware per-region semantic summaries are not yet implemented. That future
workflow should quantify subject-space signal against the warped annotation
without forcing semantic data into an instance-object model.

## Output Formats

- Large computational tables use Parquet to preserve dtypes and support fast,
  compressed programmatic access.
- Compact regional summaries use CSV for straightforward inspection and
  interchange.
- Spatial definitions and processing provenance use JSON sidecars.
- Voxel maps can be saved as NumPy arrays or NIfTI; project workflows may choose
  a narrower output contract.

## Quality Control and Validation

Available helpers cover:

- centroid image generation
- pairwise duplicate QC for deduplication
- dataset validation against declared spatial metadata
- comparison with legacy MATLAB centroid CSV files

The MATLAB CSV is a validation and compatibility format, not the canonical
`spatialsignal` storage format.

## Reference-Space Outputs

Reference-space maps are planned primarily for aligned visualization and
cross-subject comparison. The intended instance workflow is to transform
deduplicated objects into the reference space, voxelize them there, and then
derive any smoothed or boundary-corrected visualization representation.

An unmodulated aligned point map is not automatically a native-density-preserving
map after nonlinear deformation. Jacobian-adjusted representations therefore
remain a separate research lane requiring controlled validation. Canonical
regional quantification remains in subject space.
