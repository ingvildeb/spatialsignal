# spatialsignal

`spatialsignal` is a reusable Python package for deriving and analysing spatial
representations from instance or semantic segmentations produced by light-sheet
fluorescence microscopy workflows.

This is an early-stage, pre-1.0 release. The core data models and workflows are
in place, but some APIs may still evolve as the package is tested in real use.

`spatialsignal` currently supports:

- centroid point clouds from labeled instance-segmentation masks
- dense signal-support point tables from binary semantic masks
- cross-plane deduplication of repeated detections
- exact-plane, one-to-one colocalization matching between centroid point clouds
- subject-space voxelization into count maps and fraction maps
- subject-space object-to-region assignment with atlas names, counts, densities, and 2D morphology measurements
- optional hierarchy-level instance summaries using curated `atlaslevels` bundles
- explicit metadata sidecars describing space, representation, and processing
- validation against legacy MATLAB centroid CSV outputs (relevant for Kim lab members)

Large tables are stored as Parquet to preserve dtypes and support efficient
programmatic reads. Compact per-region summaries remain CSV for easy inspection.
Spatial metadata and provenance are stored in JSON sidecars.

## Core concepts

### Spatial metadata

`SpaceDefinition` describes the voxel grid occupied by a spatial dataset. It
keeps together:

- a three-letter anatomical orientation, such as `las`
- axis labels and whether voxel indices are zero- or one-based
- the grid shape
- voxel resolution in microns
- a human-readable space name

For example, define the native space occupied by a subject's segmentation
masks:

```python
from spatialsignal.models import SpaceDefinition

native_space = SpaceDefinition(
    space_name="subject_001_native",
    orientation="las",
    axis_labels=["x", "y", "z"],
    indexing="zero_based",
    units="voxel",
    shape=[640, 400, 580],
    resolution_um=[1.8, 1.8, 20.0],
)
```

Coordinates and arrays use `x, y, z` order throughout the package. Keeping the
orientation, indexing convention, shape, and resolution explicit prevents a
table or array from being interpreted in the wrong grid.

### Dataset metadata

`DatasetMetadata` combines a `SpaceDefinition` with two other pieces of
information:

- `DataRepresentation` describes what the data mean, such as raw centroid
  detections, cleaned objects, a count map, or a signal fraction map.
- `ProcessingProvenance` records the stage, source, parameters, and a compact
  processing summary when applicable.

This metadata is written beside each canonical data file as JSON. The sidecar
is part of the dataset contract rather than optional documentation.

Continuing with `native_space`, describe a raw centroid table and how it was
produced:

```python
from pathlib import Path

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    ProcessingProvenance,
)

metadata = DatasetMetadata(
    schema_name="spatialsignal.dataset_metadata",
    schema_version="0.1.0",
    space=native_space,
    representation=DataRepresentation(
        kind="point_cloud",
        representation_type="point_centroids",
    ),
    processing=ProcessingProvenance(
        stage="extract_centroids",
        source_name="masks/subject_001",
        parameters={"indexing": "zero_based"},
        summary={"n_detections": 3},
    ),
)

metadata.to_json(Path("outputs/subject_001_pointcloud_space.json"))
```

The representation says what the values mean, while the processing record says
where they came from. Both travel with the spatial definition in the saved
sidecar.

### Point clouds

`PointCloudDataset` keeps a pandas point table together with its metadata and
subject identity. It provides validation for the required columns, coordinate
bounds, indexing convention, and declared space.

The main point representations are:

- `point_centroids`: one raw centroid detection per labeled 2D instance
- `cleaned_objects`: detections believed to represent the same object across
  adjacent planes, merged into one object
- `signal_points`: every foreground voxel represented by binary semantic masks

Continuing with the `metadata` above, combine a small centroid table with its
subject identity and metadata:

```python
import pandas as pd

from spatialsignal.models import PointCloudDataset

points = pd.DataFrame(
    {
        "detection_id": [1, 2, 3],
        "seg_num": [14, 8, 21],
        "x": [120, 305, 410],
        "y": [85, 190, 250],
        "z": [12, 12, 13],
    }
)

dataset = PointCloudDataset(
    subject_name="subject_001",
    points=points,
    metadata=metadata,
)
dataset.validate()

print(dataset.summary())
```

For canonical files written by `spatialsignal`, load the Parquet table and JSON
sidecar together:

```python
from pathlib import Path

dataset = PointCloudDataset.from_files(
    table_path=Path("outputs/subject_001_pointcloud.parquet"),
    json_path=Path("outputs/subject_001_pointcloud_space.json"),
)
dataset.validate()
```

### Voxel maps

`VoxelMap` keeps a three-dimensional NumPy array together with the same
metadata contract. Its in-memory array order is `data[x, y, z]`.

Instance centroids or cleaned objects become count maps, where each value is
the number of objects assigned to an analysis voxel. Binary semantic masks
become fraction maps, where each value is the fraction of native voxels
containing signal within an analysis voxel. These representations should not be
treated as interchangeable.

For example, derive a coarser grid from `native_space` and construct a small
count map in that analysis space:

```python
import numpy as np

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    ProcessingProvenance,
    VoxelMap,
)
from spatialsignal.voxelization import make_subject_analysis_space

analysis_space = make_subject_analysis_space(
    native_space,
    analysis_resolution_um=[20.0, 20.0, 20.0],
    space_name="subject_001_analysis",
)

counts = np.zeros(tuple(analysis_space.shape), dtype=np.uint32)
counts[10, 15, 20] = 3
counts[11, 15, 20] = 1

count_metadata = DatasetMetadata(
    schema_name="spatialsignal.dataset_metadata",
    schema_version="0.1.0",
    space=analysis_space,
    representation=DataRepresentation(
        kind="voxel_map",
        representation_type="count_map",
        value_units="objects_per_voxel",
    ),
    processing=ProcessingProvenance(
        stage="voxelize_to_space",
        source_name="subject_001_objects.parquet",
        summary={"input_points": 4, "total_count": 4},
    ),
)

count_map = VoxelMap(
    subject_name="subject_001",
    data=counts,
    metadata=count_metadata,
)

print(count_map.summary())
```

In normal workflows, `voxelize_to_space` or
`voxelize_signal_masks_to_fraction_map` constructs this model and its metadata
automatically. The explicit example shows how the array, spatial grid, data
meaning, and processing history fit together.

### Subject space and reference space

The canonical quantitative outputs remain in subject space. A subject analysis
space may use a coarser resolution than the native segmentation while retaining
the same physical extent and orientation.

Reference-space maps are useful for visualization and cross-subject comparison,
but they are derived products rather than the primary source of truth for
biological quantification. Atlas-aware region quantification instead samples an
annotation volume that has been transformed into the subject's space.

## Package scope

`spatialsignal` is intended to work alongside other packages in the LSFM
analysis ecosystem:

- `atlasspace` owns template averaging, registration, transforms, and registration output folders
- `atlaslevels` owns atlas hierarchy and region metadata
- `spatialsignal` owns segmentation-derived spatial representations such as point clouds, cleaned object tables, count maps, and semantic fraction maps

## Installation

`spatialsignal` is written for Python 3.10+.

Install in editable mode from the repository root:

```bash
pip install -e .
```

For development extras:

```bash
pip install -e ".[dev]"
```

## Example instance-segmentation workflow

The following examples build on one another to turn a stack of labeled 2D
instance masks into a validated point cloud, cleaned objects, and a subject-space
count map.

### Build a centroid point cloud

Each nonzero instance label in each mask becomes one centroid detection:

```python
from pathlib import Path

from spatialsignal.pointcloud import build_pointcloud_from_masks

mask_dir = Path("masks/subject_001")
out_dir = Path("outputs")

build_pointcloud_from_masks(
    mask_dir=mask_dir,
    out_dir=out_dir,
    subject_name="subject_001",
    space_name="subject_001_native",
    orientation="las",
    resolution_um=[1.8, 1.8, 20.0],
    indexing="zero_based",
    max_workers=1,
)
```

This writes `subject_001_pointcloud.parquet` and its matching
`subject_001_pointcloud_space.json` sidecar. The stack dimensions are inferred
from the mask files.

Use `max_workers=1` in notebooks and other interactive sessions. On Windows,
keep calls with `max_workers > 1` under `if __name__ == "__main__":`.

### Merge repeated detections across planes

Load the point cloud and merge nearby detections that likely represent the same
object in adjacent sections:

```python
from spatialsignal.io import save_deduplication_outputs
from spatialsignal.models import PointCloudDataset
from spatialsignal.pointcloud import deduplicate_across_planes

dataset = PointCloudDataset.from_files(
    table_path=out_dir / "subject_001_pointcloud.parquet",
    json_path=out_dir / "subject_001_pointcloud_space.json",
)
dataset.validate()

result = deduplicate_across_planes(
    dataset.points,
    dataset.space,
    max_plane_offset=1,
    max_xy_distance_um=3.0,
    max_n_planes=2,
)

dedup_paths = save_deduplication_outputs(dataset, result, out_dir)
print(f"{len(dataset.points)} detections -> {len(result.objects)} objects")
```

`result` contains the cleaned object table, the raw-detection membership table,
and the accepted cross-plane edges. The saved object sidecar also records the
deduplication parameters and summary.

### Voxelize cleaned objects

Create a coarser analysis grid covering the same subject and aggregate one count
per cleaned object:

```python
from spatialsignal.io import save_voxel_map_outputs
from spatialsignal.voxelization import make_subject_analysis_space, voxelize_to_space

objects = PointCloudDataset.from_files(
    table_path=dedup_paths.objects_table,
    json_path=dedup_paths.objects_json,
    subject_name="subject_001",
)

analysis_space = make_subject_analysis_space(
    objects.space,
    analysis_resolution_um=[20.0, 20.0, 20.0],
    space_name="subject_001_analysis",
)
count_map = voxelize_to_space(objects, analysis_space)
count_paths = save_voxel_map_outputs(
    count_map,
    out_dir,
    name_suffix="count_map",
    formats=("nifti",),
)

print(count_map.summary())
print(count_paths.nifti_path)
```

## Example semantic-segmentation workflow

For binary semantic masks, voxelize the masks directly into a fraction map. This
avoids materializing a potentially very large signal-point table:

```python
from pathlib import Path

from spatialsignal.io import find_mask_files, save_voxel_map_outputs
from spatialsignal.models import SpaceDefinition
from spatialsignal.voxelization import (
    make_subject_analysis_space,
    voxelize_signal_masks_to_fraction_map,
)

mask_files = find_mask_files(Path("masks/subject_001_signal"))
native_space = SpaceDefinition.from_mask_files(
    space_name="subject_001_native",
    orientation="las",
    resolution_um=[1.8, 1.8, 20.0],
    indexing="zero_based",
    mask_files=mask_files,
)
analysis_space = make_subject_analysis_space(
    native_space,
    analysis_resolution_um=[20.0, 20.0, 20.0],
    space_name="subject_001_analysis",
)
fraction_map = voxelize_signal_masks_to_fraction_map(
    mask_files,
    native_space,
    analysis_space,
    subject_name="subject_001",
)
fraction_paths = save_voxel_map_outputs(
    fraction_map,
    Path("outputs"),
    name_suffix="fraction_map",
    formats=("nifti",),
)

print(fraction_map.summary())
print(fraction_paths.nifti_path)
```

## Atlas-aware region quantification

Cleaned objects can be assigned to a subject-space annotation produced by an
`atlasspace` registration. Region summaries retain the annotation's original
`region_id` and can be enriched with canonical Allen IDs, acronyms, names, and
colors through `atlaslevels`.

Start from the cleaned-object table written by the deduplication workflow, then
load the subject-space annotation and build one summary row per atlas region:

```python
from pathlib import Path

from spatialsignal.integration import load_label_volume
from spatialsignal.models import PointCloudDataset
from spatialsignal.quantification import quantify_objects_by_region

objects = PointCloudDataset.from_files(
    table_path=Path("outputs/subject_001_objects.parquet"),
    json_path=Path("outputs/subject_001_objects_space.json"),
    subject_name="subject_001",
)

annotation = load_label_volume(
    Path("outputs/subject_001_registration/annotation_WarpedSegmentation.nii.gz")
)

result = quantify_objects_by_region(
    objects,
    annotation,
    ontology_preset="allen_ccfv3",
    region_id_space="kimlab16bit",  # Use "allen" for an Allen-ID annotation.
    output_csv=Path("outputs/subject_001_region_summary.csv"),
)

print(result.assigned_objects["assignment_status"].value_counts())
print(
    result.region_summary[
        ["region_acronym", "region_name", "object_count", "object_density_per_mm3"]
    ]
)
print(result.summary_csv)
print(result.qc_png)
```

`load_label_volume` reads the annotation array and derives its sampling grid
from the NIfTI shape and affine. Here, the annotation has already been
transformed into subject space by `atlasspace`; loading the surrounding
registration folder is not required. Its optional `space_name` is only a
provenance label and defaults to the NIfTI filename stem.

`quantify_objects_by_region` expresses the cleaned-object coordinates in its
sampling grid, assigns region IDs, adds atlas names, and returns both the
object-level assignments and regional summary. Passing `output_csv` also writes
the summary to the requested CSV path and automatically writes the matching
`subject_001_quantification_qc.png`.

Objects sampling annotation ID `0` remain in `result.assigned_objects` with
`assignment_status == "background"`, but background is excluded from
`result.region_summary` by default. The annotation therefore provides all
information required for regional quantification; a separate brain mask is not
needed. Set `region_id_space` to the ID namespace actually stored in the
annotation.

See the [workflow guide](docs/workflow.md) and
[region-quantification example](examples/quantify_objects_by_region.py) for the
alternative registration-folder-to-region-report workflow, which discovers the
annotation path from an `atlasspace` output folder.

### Automatic quantification QC

When quantification outputs are written, the QC PNG is generated automatically.
The plotting helper remains available when regenerating QC from an in-memory or
previously saved assigned-object dataset:

```python
from spatialsignal.quantification import write_region_quantification_qc

qc_path = write_region_quantification_qc(
    result,
    annotation,
    Path("outputs/subject_001_quantification_qc_review.png"),
)
print(qc_path)
```

The PNG contains three whole-volume point-count projections with the annotation
outline, plus three representative annotation slices with regional boundaries
and assigned points. Background-assigned objects are highlighted in red on the
slice panels, where their depth is unambiguous. A summary reports assigned,
background, and out-of-bounds percentages. The projections accumulate directly
into 2D histograms, so this does not create a dense 3D count map.

## Packaged templates

Canonical starter TOMLs live in the installable package under
`spatialsignal.config_templates`.

Scaffold a local config through the Python API:

```python
from pathlib import Path

from spatialsignal.config_templates import scaffold_template

scaffold_template("build_pointcloud", Path("build_pointcloud_local.toml"))
```

Starter templates currently include:

- `build_pointcloud`
- `build_signal_points`
- `colocalization`
- `voxelize_signal_masks`

Starter configs use zero-based indexing, explicit `[space]` metadata, and
single-quoted Windows paths.

## Example runners

Runnable wrappers in `examples/` demonstrate local-file and config-driven use:

- `examples/build_pointcloud.py`
- `examples/build_signal_points.py`
- `examples/colocalize_pointclouds.py`
- `examples/deduplicate_across_planes.py`
- `examples/quantify_objects_by_region.py`
- `examples/test_python_workflow.py`
- `examples/test_region_quantification_workflow.py`
- `examples/validate_against_matlab.py`
- `examples/validate_pointcloud_dataset.py`
- `examples/voxelize_pointcloud.py`
- `examples/voxelize_signal_masks.py`
- `examples/scaffold_config.py`

These are convenience wrappers and testing helpers rather than the canonical
package interface.

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

## License

This project is released under the terms of the [LICENSE](LICENSE).
