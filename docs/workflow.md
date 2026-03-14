# Workflow

## Milestone 1

The first implementation milestone focuses on a single task:

1. Read Cellpose mask images from a directory.
2. Assign sequential slice indices from naturally sorted filenames.
3. Extract one centroid per labeled object.
4. Export a clean 3D point-cloud CSV.
5. Optionally write centroid QC images.
6. Validate the output against the legacy MATLAB centroid CSV.

## In Scope

- Cellpose mask ingestion from `masks_*.tif*`
- Centroid extraction from labeled 2D masks
- Canonical point-cloud CSV export with explicit spatial column names
- Optional centroid QC image output
- Comparison against the legacy MATLAB centroid CSV
- Unit tests on small synthetic masks

## Out of Scope

- ANTs registration
- Atlas transforms
- Reorientation into anatomical atlas spaces
- Voxelized density maps
- Group statistics or hotspot detection

## Planned Next Steps

After milestone 1 is stable, the next additions should be:

1. Point-cloud to voxel count-map conversion
2. Reorientation and coordinate-space utilities
3. Support for mapping into age-specific reference spaces and CCFv3
