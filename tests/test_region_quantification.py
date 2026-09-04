"""Tests for the high-level atlas-region quantification workflow."""

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
from spatialsignal.quantification import (
    RegionQuantificationResult,
    quantify_objects_by_region,
)


def _space() -> SpaceDefinition:
    return SpaceDefinition(
        space_name="subject_annotation",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[3, 1, 1],
        resolution_um=[10.0, 10.0, 10.0],
    )


def _objects() -> PointCloudDataset:
    space = _space()
    return PointCloudDataset(
        subject_name="subject_001",
        points=pd.DataFrame(
            {
                "object_id": [1, 2, 3],
                "x": [0, 1, 2],
                "y": [0, 0, 0],
                "z": [0, 0, 0],
                "x_float": [0.0, 1.0, 2.0],
                "y_float": [0.0, 0.0, 0.0],
                "z_float": [0.0, 0.0, 0.0],
                "n_detections": [1, 1, 2],
                "n_planes": [1, 1, 2],
            }
        ),
        metadata=DatasetMetadata(
            schema_name="spatialsignal.dataset_metadata",
            schema_version="0.1.0",
            space=space,
            representation=DataRepresentation(
                kind="point_cloud",
                representation_type="cleaned_objects",
            ),
        ),
    )


def _annotation() -> LabelVolume:
    return LabelVolume(
        path=Path("subject_annotation.nii.gz"),
        data=np.array([[[68]], [[0]], [[667]]], dtype=np.int32),
        space=_space(),
    )


def _hemisphere() -> LabelVolume:
    return LabelVolume(
        path=Path("subject_hemispheres.nii.gz"),
        data=np.array([[[1]], [[2]], [[2]]], dtype=np.uint8),
        space=_space(),
    )


def test_quantify_objects_by_region_returns_named_summary_and_writes_csv(
    tmp_path: Path,
) -> None:
    output_csv = tmp_path / "subject_001_region_summary.csv"

    result = quantify_objects_by_region(
        _objects(),
        _annotation(),
        ontology_preset="allen_ccfv3",
        region_id_space="allen",
        output_csv=output_csv,
    )

    assert isinstance(result, RegionQuantificationResult)
    assert result.summary_csv == output_csv
    assert result.qc_png == tmp_path / "subject_001_quantification_qc.png"
    assert result.qc_png.is_file()
    assert list(result.assigned_objects["region_id"]) == [68, 0, 667]
    assert list(result.assigned_objects["assignment_status"]) == [
        "assigned",
        "background",
        "assigned",
    ]
    assert set(result.region_summary["region_id"]) == {68, 667}
    assert set(result.region_summary["region_name"]) == {
        "Frontal pole, layer 1",
        "Frontal pole, layer 2/3",
    }
    assert output_csv.exists()
    pd.testing.assert_frame_equal(
        pd.read_csv(output_csv),
        result.region_summary,
        check_dtype=False,
    )


def test_quantify_objects_by_region_can_include_annotation_background() -> None:
    result = quantify_objects_by_region(
        _objects(),
        _annotation(),
        ontology_preset="allen_ccfv3",
        region_id_space="allen",
        include_background=True,
    )

    assert result.summary_csv is None
    assert result.qc_png is None
    assert set(result.region_summary["region_id"]) == {0, 68, 667}
    background = result.region_summary.loc[result.region_summary["region_id"] == 0].iloc[0]
    assert background["region_name"] == "Background"
    assert background["bilateral_object_count"] == 1


def test_quantify_objects_by_region_reports_bilateral_left_and_right_metrics() -> None:
    result = quantify_objects_by_region(
        _objects(),
        _annotation(),
        hemisphere=_hemisphere(),
        ontology_preset="allen_ccfv3",
        region_id_space="allen",
    )

    assert result.assigned_objects["hemisphere"].astype("string").tolist() == [
        "left",
        "right",
        "right",
    ]
    region_68 = result.region_summary.loc[
        result.region_summary["region_id"] == 68
    ].iloc[0]
    assert region_68["bilateral_object_count"] == 1
    assert region_68["left_object_count"] == 1
    assert region_68["right_object_count"] == 0
    assert region_68["bilateral_region_voxels"] == 1
    assert region_68["left_region_voxels"] == 1
    assert region_68["right_region_voxels"] == 0
    assert region_68["left_object_density_per_mm3"] == 1_000_000.0
    assert np.isnan(region_68["right_object_density_per_mm3"])


def test_quantify_objects_by_region_rejects_unassigned_annotated_hemisphere() -> None:
    hemisphere = _hemisphere()
    hemisphere.data[2, 0, 0] = 0

    with pytest.raises(ValueError, match="Every annotated voxel"):
        quantify_objects_by_region(
            _objects(),
            _annotation(),
            hemisphere=hemisphere,
        )


def test_quantify_objects_by_region_requires_csv_output_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_csv must have a .csv suffix"):
        quantify_objects_by_region(
            _objects(),
            _annotation(),
            output_csv=tmp_path / "subject_001_region_summary.txt",
        )


def test_quantify_objects_by_region_can_disable_automatic_qc(tmp_path: Path) -> None:
    result = quantify_objects_by_region(
        _objects(),
        _annotation(),
        output_csv=tmp_path / "subject_001_region_summary.csv",
        generate_qc=False,
    )

    assert result.summary_csv is not None
    assert result.qc_png is None
    assert not (tmp_path / "subject_001_quantification_qc.png").exists()


def test_quantify_objects_by_region_rejects_raw_detection_pointcloud() -> None:
    objects = _objects()
    objects.metadata.representation.representation_type = "point_centroids"

    with pytest.raises(ValueError, match="requires a cleaned-object point cloud"):
        quantify_objects_by_region(objects, _annotation())
