from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from spatialsignal.integration import transform_pointcloud_to_reference_space
from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
)
from spatialsignal.voxelization import build_nifti_ras_affine, voxelize_to_space


def _space(name: str) -> SpaceDefinition:
    return SpaceDefinition(
        space_name=name,
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[4, 4, 4],
        resolution_um=[20.0, 20.0, 20.0],
        affine_ras_mm=build_nifti_ras_affine(
            SpaceDefinition(
                space_name=name,
                orientation="las",
                axis_labels=["x", "y", "z"],
                indexing="zero_based",
                units="voxel",
                shape=[4, 4, 4],
                resolution_um=[20.0, 20.0, 20.0],
            )
        ).tolist(),
    )


def _registration_dir(tmp_path: Path) -> Path:
    fixed = _space("subject")
    moving = _space("template")
    for filename, space in (("fixed.nii.gz", fixed), ("moving.nii.gz", moving)):
        image = nib.Nifti1Image(
            np.zeros(tuple(space.shape), dtype=np.uint8),
            np.asarray(space.affine_ras_mm),
        )
        image.header.set_xyzt_units("mm")
        nib.save(image, str(tmp_path / filename))
    manifest_space = lambda space: {
        "space_name": space.space_name,
        "orientation": space.orientation,
        "axis_labels": space.axis_labels,
        "units": space.units,
        "resolution_um": space.resolution_um,
        "shape": space.shape,
    }
    manifest = {
        "schema_version": 1,
        "success": True,
        "preset_name": "identity",
        "fixed_image": {
            "image": "native.nii.gz",
            "normalized_image": "fixed.nii.gz",
            "space": manifest_space(fixed),
        },
        "moving_image": {
            "image": "template.nii.gz",
            "normalized_image": "moving.nii.gz",
            "space": manifest_space(moving),
        },
        "effective_fixed_space": manifest_space(fixed),
        "effective_moving_space": manifest_space(moving),
        "forward_transforms": ["forward.mat"],
        "inverse_transforms": ["inverse.mat"],
        "transformed_segmentations": {},
    }
    (tmp_path / "registration_result.json").write_text(json.dumps(manifest))
    return tmp_path


def _dataset() -> PointCloudDataset:
    space = _space("native")
    points = pd.DataFrame(
        {
            "object_id": [10, 11, 12, 13],
            "seg_num": [1, 2, 3, 4],
            "x": [0, 1, 2, 3],
            "y": [0, 1, 2, 3],
            "z": [0, 1, 2, 3],
            "x_float": [0.25, 1.25, 2.25, 3.0],
            "y_float": [0.0, 1.0, 2.0, 3.0],
            "z_float": [0.0, 1.0, 2.0, 3.0],
            "volume_voxels": [5, 6, 7, 8],
        }
    )
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=space,
        representation=DataRepresentation(
            kind="point_cloud", representation_type="cleaned_objects"
        ),
        processing=ProcessingProvenance(stage="test", source_name="objects.parquet"),
    )
    return PointCloudDataset("SUBJECT", points, metadata)


def test_reference_transform_classifies_bounds_and_preserves_columns(
    tmp_path: Path, monkeypatch
) -> None:
    registration_dir = _registration_dir(tmp_path)
    transformed = np.array(
        [
            [1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2],
            [np.nan, 2.0, 2.0],
            [9.0, 3.0, 3.0],
        ]
    )
    monkeypatch.setattr(
        "spatialsignal.integration.reference_space._transform_registration_indices",
        lambda source, registration_dir, *, chunk_size: transformed,
    )

    result = transform_pointcloud_to_reference_space(
        _dataset(), registration_dir, chunk_size=2
    )

    assert result.input_count == 4
    assert result.transformed_count == 4
    assert result.in_bounds_count == 2
    assert result.non_finite_count == 1
    assert result.out_of_bounds_count == 1
    assert result.rejected_points["rejection_reason"].tolist() == [
        "non_finite",
        "out_of_bounds",
    ]
    assert result.dataset.points["object_id"].tolist() == [10, 11]
    assert result.dataset.points["volume_voxels"].tolist() == [5, 6]
    assert "native_x_float" in result.dataset.points
    assert result.dataset.space.space_name == "template"
    np.testing.assert_array_equal(
        np.asarray(result.dataset.space.affine_ras_mm),
        np.asarray(nib.load(str(registration_dir / "moving.nii.gz")).affine),
    )
    processing = result.dataset.metadata.processing
    assert processing is not None
    assert processing.parameters["modulation"] == "none"
    assert processing.summary["input_points"] == 4

    count_map = voxelize_to_space(
        result.dataset,
        result.dataset.space,
        source_name="reference_points.parquet",
        processing_parameters={
            "space_kind": "reference",
            "modulation": "none",
            "point_weight": 1,
        },
    )
    assert count_map.data.shape == (4, 4, 4)
    assert int(count_map.data.sum()) == 2
    assert int(count_map.data.max()) == 2
    assert count_map.metadata.processing.parameters["space_kind"] == "reference"
