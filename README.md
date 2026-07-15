# spatialsignal

`spatialsignal` is a reusable Python package for segmentation-derived spatial
representations from light-sheet fluorescence microscopy workflows.

The package is designed as the downstream quantification layer in the LSFM
ecosystem. It focuses on building validated subject-space point clouds and
voxel maps, writing explicit spatial metadata, and supporting later atlas-aware
quantification.

`spatialsignal` currently supports:

- centroid point clouds from labeled instance-segmentation masks
- dense signal-support point tables from binary semantic masks
- cross-plane deduplication of repeated detections
- subject-space voxelization into count maps and fraction maps
- subject-space object-to-region assignment with atlas names, counts, densities, and morphology
- explicit metadata sidecars describing space, representation, and processing
- validation against legacy MATLAB centroid CSV outputs

Large computational tables are stored as Parquet to preserve dtypes and support
efficient programmatic reads. Compact per-region summaries remain CSV for easy
inspection, and spatial metadata and provenance remain in JSON sidecars.

## Package scope

`spatialsignal` is intended to work alongside other packages in the LSFM
analysis ecosystem:

- `atlasspace` owns template averaging, registration, transforms, and registration output folders
- `atlaslevels` owns atlas hierarchy and region metadata
- `spatialsignal` owns segmentation-derived spatial representations such as point clouds, cleaned object tables, count maps, and semantic fraction maps

The canonical quantitative outputs are intended to remain in subject space.
Reference-space maps are useful for visualization and cross-subject comparison,
but they are not the primary source of truth for biological quantification.

## Installation

Install in editable mode from the repository root:

```text
pip install -e .
```

For development extras:

```text
pip install -e ".[dev]"
```

## Minimal Python Workflow

A minimal subject-space workflow looks like this:

```python
from pathlib import Path

from spatialsignal.io import save_deduplication_outputs, save_voxel_map_outputs
from spatialsignal.pointcloud import (
    PointCloudDataset,
    build_pointcloud_from_masks,
    deduplicate_across_planes,
)
from spatialsignal.voxelization import (
    make_subject_analysis_space,
    voxelize_to_space,
)

MASK_DIR = Path("masks")
OUT_DIR = Path("outputs")
SUBJECT_NAME = "subject_001"
SPACE_NAME = "subject_space"
ORIENTATION = "las"
RESOLUTION_UM = [1.8, 1.8, 20.0]
INDEXING = "zero_based"
MAX_WORKERS = 1


if __name__ == "__main__":
    build_pointcloud_from_masks(
        mask_dir=MASK_DIR,
        out_dir=OUT_DIR,
        subject_name=SUBJECT_NAME,
        space_name=SPACE_NAME,
        orientation=ORIENTATION,
        resolution_um=RESOLUTION_UM,
        indexing=INDEXING,
        max_workers=MAX_WORKERS,
    )

    dataset = PointCloudDataset.from_files(
        table_path=OUT_DIR / f"{SUBJECT_NAME}_pointcloud.parquet",
        json_path=OUT_DIR / f"{SUBJECT_NAME}_pointcloud_space.json",
    )
    dataset.validate()

    result = deduplicate_across_planes(
        dataset.points,
        dataset.space,
        max_plane_offset=1,
        max_xy_distance_um=3.0,
        max_n_planes=2,
    )
    dedup_paths = save_deduplication_outputs(dataset, result, OUT_DIR)
    objects = PointCloudDataset.from_files(
        dedup_paths.objects_table,
        dedup_paths.objects_json,
        subject_name=SUBJECT_NAME,
    )

    analysis_space = make_subject_analysis_space(
        objects.space,
        analysis_resolution_um=[20.0, 20.0, 20.0],
        space_name="subject_analysis_space",
    )
    count_map = voxelize_to_space(objects, analysis_space)
    count_paths = save_voxel_map_outputs(
        count_map,
        OUT_DIR,
        name_suffix="count_map",
        formats=("nifti",),
    )

    print(len(result.objects))
    print(count_map.data.shape)
    print(count_paths.nifti_path)
```

Use `max_workers=1` in notebooks and other interactive sessions. On Windows, if
you want `max_workers > 1`, keep the process-pool execution under
`if __name__ == "__main__":`.

This package convention is:

- keep subject-space outputs as the canonical quantitative lane
- use explicit metadata sidecars to define orientation, resolution, and indexing
- treat example runners as convenience wrappers around the core Python API

## Packaged Templates

Canonical starter TOMLs live in the installable package under
`spatialsignal.config_templates`.

If you want a local config file to edit, you can scaffold one through the
Python API:

```python
from pathlib import Path

from spatialsignal.config_templates import scaffold_template

scaffold_template("build_pointcloud", Path("build_pointcloud_local.toml"))
```

Starter templates currently include:

- `build_pointcloud`
- `build_signal_points`
- `voxelize_signal_masks`

Starter configs use:

- `indexing = "zero_based"`
- explicit `[space]` metadata
- single-quoted Windows paths

## Example Runners

The repo also includes runnable wrappers in `examples/` that demonstrate how to
use the API on local files:

- `examples/build_pointcloud.py`
- `examples/build_signal_points.py`
- `examples/deduplicate_across_planes.py`
- `examples/quantify_objects_by_region.py`
- `examples/test_python_workflow.py`
- `examples/test_region_quantification_workflow.py`
- `examples/validate_against_matlab.py`
- `examples/validate_pointcloud_dataset.py`
- `examples/voxelize_pointcloud.py`
- `examples/voxelize_signal_masks.py`
- `examples/scaffold_config.py`

These are intended as examples and testing helpers rather than the canonical
package interface.

Region-report workflows use `atlaslevels` automatically. Set the annotation ID
namespace to `allen` or `kimlab16bit`; the saved report retains the annotation's
original `region_id` and adds the canonical Allen ID, acronym, name, and color.
See the [workflow guide](docs/workflow.md) and
[region-quantification example](examples/quantify_objects_by_region.py) for the
complete registration-folder-to-region-report workflow.

## Documentation

- [Workflow guide](docs/workflow.md): supported workflows and package boundaries
- [Coordinate systems](docs/coordinate_systems.md): coordinate conventions and map semantics
- [Legacy MATLAB relationship](docs/legacy_matlab_relationship.md): compatibility notes
- [Roadmap](docs/roadmap.md): internal development status and planned work

## Repository layout

```text
src/        Reusable package code
src/spatialsignal/config_templates/  Canonical packaged starter TOMLs
examples/   Example wrappers and API-usage patterns
docs/       User-facing and planning documentation
tests/      Unit tests
```
