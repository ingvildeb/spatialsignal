from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialsignal.io import save_pointcloud_dataset
from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.pointcloud import filter_detections_by_area


def _dataset(points: pd.DataFrame | None = None) -> PointCloudDataset:
    if points is None:
        points = pd.DataFrame(
            [
                {
                    "detection_id": 11,
                    "seg_num": 1,
                    "x": 2,
                    "y": 3,
                    "z": 0,
                    "area_px": 5,
                },
                {
                    "detection_id": 12,
                    "seg_num": 2,
                    "x": 4,
                    "y": 5,
                    "z": 1,
                    "area_px": 10,
                },
                {
                    "detection_id": 13,
                    "seg_num": 3,
                    "x": 6,
                    "y": 7,
                    "z": 2,
                    "area_px": 11,
                },
            ]
        )
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native_masks",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="zero_based",
            units="voxel",
            shape=[20, 20, 3],
            resolution_um=[2.0, 3.0, 20.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    return PointCloudDataset("subject_1", points, metadata)


def test_filter_detections_by_area_uses_calibrated_inclusive_boundary() -> None:
    dataset = _dataset()
    source_points = dataset.points.copy(deep=True)
    source_metadata = deepcopy(dataset.metadata.to_dict())

    result = filter_detections_by_area(
        dataset,
        maximum_area_um2=60.0,
        source_name="subject_1_pointcloud.parquet",
    )

    assert result.accepted.points["detection_id"].tolist() == [11, 12]
    assert result.rejected.points["detection_id"].tolist() == [13]
    assert result.accepted.points["area_um2"].tolist() == [30.0, 60.0]
    assert result.rejected.points["area_um2"].tolist() == [66.0]
    assert result.summary == {
        "source_detections": 3,
        "accepted_detections": 2,
        "rejected_detections": 1,
        "maximum_area_um2": 60.0,
        "pixel_area_um2": 6.0,
        "accepted_rule": "area_um2 <= maximum_area_um2",
    }
    assert (
        result.accepted.metadata.representation.representation_type
        == "area_filtered_point_centroids"
    )
    processing = result.accepted.metadata.processing
    assert processing is not None
    assert processing.stage == "filter_detections_by_area"
    assert processing.source_name == "subject_1_pointcloud.parquet"
    assert processing.parameters is not None
    assert processing.parameters["population"] == "accepted"
    assert processing.summary == {
        "source_detections": 3,
        "accepted_detections": 2,
        "rejected_detections": 1,
    }
    pd.testing.assert_frame_equal(dataset.points, source_points)
    assert dataset.metadata.to_dict() == source_metadata


def test_filter_detections_by_area_resolves_xy_calibration_by_axis_label() -> None:
    dataset = _dataset()
    dataset.metadata.space.axis_labels = ["z", "x", "y"]
    dataset.metadata.space.shape = [3, 20, 20]
    dataset.metadata.space.resolution_um = [20.0, 2.0, 3.0]

    result = filter_detections_by_area(dataset, maximum_area_um2=60.0)

    assert result.pixel_area_um2 == 6.0
    assert result.accepted.points["detection_id"].tolist() == [11, 12]


@pytest.mark.parametrize("maximum", [0.0, -1.0, np.nan, np.inf])
def test_filter_detections_by_area_rejects_invalid_threshold(maximum: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        filter_detections_by_area(_dataset(), maximum_area_um2=maximum)


def test_filter_detections_by_area_requires_valid_detection_areas() -> None:
    missing = _dataset().points.drop(columns="area_px")
    with pytest.raises(ValueError, match="missing the detection-area column"):
        filter_detections_by_area(_dataset(missing), maximum_area_um2=60.0)

    invalid = _dataset().points.copy()
    invalid.loc[1, "area_px"] = np.nan
    with pytest.raises(ValueError, match=r"invalid detection IDs: \[12\]"):
        filter_detections_by_area(_dataset(invalid), maximum_area_um2=60.0)


def test_save_pointcloud_dataset_round_trips_filtered_population(
    tmp_path: Path,
) -> None:
    result = filter_detections_by_area(_dataset(), maximum_area_um2=60.0)
    table_path = tmp_path / "accepted.parquet"
    metadata_path = tmp_path / "accepted_space.json"

    paths = save_pointcloud_dataset(
        result.accepted,
        table_path,
        metadata_path,
    )
    loaded = PointCloudDataset.from_files(
        paths.table_path,
        paths.metadata_path,
        subject_name="subject_1",
    )

    pd.testing.assert_frame_equal(loaded.points, result.accepted.points)
    assert loaded.metadata.to_dict() == result.accepted.metadata.to_dict()
