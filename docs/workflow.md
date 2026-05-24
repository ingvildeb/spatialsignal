# Workflow

## Milestone 1

The first implementation milestone focuses on a single task:

1. Read Cellpose mask images from a directory.
2. Assign sequential slice indices from naturally sorted filenames.
3. Extract one centroid per labeled object.
4. Export a clean 3D point-cloud CSV.
5. Export a space metadata JSON describing the point-cloud space.
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
