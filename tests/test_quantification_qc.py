"""Tests for optional visual QC of regional quantification."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialsignal.integration import LabelVolume
from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.quantification import write_region_quantification_qc


pytest.importorskip("matplotlib")


def test_write_region_quantification_qc_from_saved_dataset(tmp_path: Path) -> None:
    space = SpaceDefinition(
        space_name="subject_annotation",
        orientation="lsp",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[12, 10, 8],
        resolution_um=[20.0, 20.0, 20.0],
    )
    annotation_data = np.zeros(space.shape, dtype=np.uint16)
    annotation_data[2:10, 2:8, 1:7] = 68
    annotation = LabelVolume(
        path=tmp_path / "annotation.nii.gz",
        data=annotation_data,
        space=space,
    )
    points = pd.DataFrame(
        {
            "x": [3, 5, 8, 0, 20],
            "y": [3, 5, 6, 0, 5],
            "z": [2, 4, 5, 0, 4],
            "region_id": [68, 68, 68, 0, -1],
            "assignment_status": [
                "assigned",
                "assigned",
                "assigned",
                "background",
                "out_of_bounds",
            ],
        }
    )
    dataset = PointCloudDataset(
        subject_name="subject_001",
        points=points,
        metadata=DatasetMetadata(
            schema_name="spatialsignal.dataset_metadata",
            schema_version="0.1.0",
            space=space,
            representation=DataRepresentation(
                kind="point_cloud",
                representation_type="objects_with_region_ids",
            ),
        ),
    )
    output_path = tmp_path / "subject_001_quantification_qc.png"

    returned_path = write_region_quantification_qc(
        dataset,
        annotation,
        output_path,
        max_points_per_slice=10,
        dpi=72,
    )

    assert returned_path == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_write_region_quantification_qc_requires_png_output(tmp_path: Path) -> None:
    space = SpaceDefinition(
        space_name="subject",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 2, 2],
        resolution_um=[20.0, 20.0, 20.0],
    )
    dataset = PointCloudDataset(
        subject_name="subject",
        points=pd.DataFrame({"x": [0], "y": [0], "z": [0], "region_id": [0]}),
        metadata=DatasetMetadata(
            schema_name="spatialsignal.dataset_metadata",
            schema_version="0.1.0",
            space=space,
            representation=DataRepresentation(
                kind="point_cloud",
                representation_type="objects_with_region_ids",
            ),
        ),
    )
    annotation = LabelVolume(
        path=tmp_path / "annotation.nii.gz",
        data=np.zeros(space.shape, dtype=np.uint16),
        space=space,
    )

    with pytest.raises(ValueError, match=".png suffix"):
        write_region_quantification_qc(
            dataset,
            annotation,
            tmp_path / "qc.jpg",
        )
