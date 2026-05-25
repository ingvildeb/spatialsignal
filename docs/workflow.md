# Workflow

## Milestone 1

The first implementation milestone focuses on a single task:

1. Read Cellpose mask images from a directory.
2. Assign sequential slice indices from naturally sorted filenames.
3. Extract one centroid per labeled object.
4. Export a clean 3D point-cloud CSV.
5. Export a metadata JSON describing the point-cloud space and representation type.
6. Write centroid QC images.
7. Validate the output against the legacy MATLAB centroid CSV.

## In Scope

- Cellpose mask ingestion from `masks_*.tif*`
- Centroid extraction from labeled 2D masks
- Canonical point-cloud CSV export with explicit `x,y,z` coordinates
- Standard per-object mask-derived properties stored alongside the points
- Point-cloud space JSON export
- Centroid QC image output
- Comparison against the legacy MATLAB centroid CSV
- Unit tests on small synthetic masks

## Out of Scope

- ANTs registration
- Atlas transforms
- Common-space registration of point clouds
- Voxelized density maps
- Group statistics or hotspot detection

## Updated Roadmap

After milestone 1, the planned next milestones are:

1. Point-cloud space and metadata utilities
2. General voxelization utilities operating within the current point-cloud space
3. Coordinate transforms and support for mapping point clouds into age-specific reference spaces and CCFv3
4. Analysis-ready outputs such as group-level density maps and hotspot-oriented summaries

## Deduplicate Across Planes

The next processing stage after raw point-cloud build is cross-plane
deduplication of likely repeated cell-body detections.

### Purpose

- input: raw detection point cloud
- output: cleaned object point cloud
- goal: merge detections that likely represent the same biological cell across
  nearby planes only

This stage is intentionally separate from centroid extraction. It does not try
to split touching cells within one plane; for Cellpose-based workflows that is
treated as an upstream instance-segmentation problem.

### Current Method

The current implementation:

1. Starts from the raw detection point cloud.
2. Compares detections only across forward neighboring planes up to
   `max_plane_offset`.
3. Uses lateral distance in physical units (`max_xy_distance_um`) to decide
   whether two detections can be linked.
4. Builds connected components from accepted links.
5. Optionally enforces a biological cap with `max_n_planes`.
6. Aggregates each component into one cleaned object row.

### Current Outputs

The deduplication stage writes:

- `{subject_name}_objects.csv`
- `{subject_name}_objects_space.json`
- `{subject_name}_object_membership.csv`
  - detection-to-object membership mapping from raw `detection_id` values to
    cleaned `object_id` values

Optional debug output:

- `{subject_name}_object_edges.csv`
  - accepted pairwise cross-plane links used internally to build the cleaned
    objects

The cleaned objects JSON keeps the same spatial frame as the raw point cloud
and also records a `processing` block containing:

- `stage`
- `source_pointcloud_csv`
- `parameters`
- `summary`

### Optional Pairwise QC

The deduplication script can optionally create mask-based pairwise QC images
for a few sampled adjacent plane pairs. These QC images:

- color objects green if they were linked across that exact plane pair
- color objects red if they were not linked across that exact plane pair
- require the original `masks_*.tif*` files to be provided explicitly
- validate mask/point-cloud alignment before writing output

QC files are written into a `pair_qc/` subfolder. The current visualization is
most directly interpretable when the deduplicated workflow uses a biological
cap of at most two contributing planes per cleaned object.
