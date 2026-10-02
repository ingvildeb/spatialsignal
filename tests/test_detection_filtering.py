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
from spatialsignal.pointcloud import (
    CONDITIONAL_REJECTION_REASON,
    HARD_AREA_REJECTION_REASON,
    DetectionFilterCondition,
    filter_detections_by_area,
    filter_detections_by_morphology,
)


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
                    "eccentricity": 0.20,
                    "mean_intensity": 120.0,
                },
                {
                    "detection_id": 12,
                    "seg_num": 2,
                    "x": 4,
                    "y": 5,
                    "z": 1,
                    "area_px": 10,
                    "eccentricity": 0.97,
                    "mean_intensity": 80.0,
                },
                {
                    "detection_id": 13,
                    "seg_num": 3,
                    "x": 6,
                    "y": 7,
                    "z": 2,
                    "area_px": 11,
                    "eccentricity": 0.99,
                    "mean_intensity": 120.0,
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
        "rejected_hard_area": 1,
        "rejected_conditional": 0,
        "hard_area_threshold_um2": 60.0,
        "conditional_area_threshold_um2": None,
        "conditions": [],
        "condition_combination": "any",
        "pixel_area_um2": 6.0,
        "accepted_rule": "area_um2 <= 60",
    }
    assert result.maximum_area_um2 == 60.0
    assert result.rejected.points["rejection_reason"].tolist() == [
        HARD_AREA_REJECTION_REASON
    ]
    assert (
        result.accepted.metadata.representation.representation_type
        == "morphology_filtered_point_centroids"
    )
    processing = result.accepted.metadata.processing
    assert processing is not None
    assert processing.stage == "filter_detections_by_morphology"
    assert processing.source_name == "subject_1_pointcloud.parquet"
    assert processing.parameters is not None
    assert processing.parameters["population"] == "accepted"
    assert processing.summary == {
        "source_detections": 3,
        "accepted_detections": 2,
        "rejected_detections": 1,
        "rejected_hard_area": 1,
        "rejected_conditional": 0,
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


def test_composite_filter_separates_hard_and_conditional_rejections() -> None:
    points = pd.DataFrame(
        [
            {
                "detection_id": 11,
                "seg_num": 1,
                "x": 2,
                "y": 3,
                "z": 0,
                "area_px": 8,
                "eccentricity": 1.0,
            },
            {
                "detection_id": 12,
                "seg_num": 2,
                "x": 4,
                "y": 5,
                "z": 1,
                "area_px": 9,
                "eccentricity": 0.96,
            },
            {
                "detection_id": 13,
                "seg_num": 3,
                "x": 6,
                "y": 7,
                "z": 1,
                "area_px": 9,
                "eccentricity": 0.965,
            },
            {
                "detection_id": 14,
                "seg_num": 4,
                "x": 8,
                "y": 9,
                "z": 2,
                "area_px": 11,
                "eccentricity": 0.10,
            },
        ]
    )
    source = _dataset(points)
    source_points = source.points.copy(deep=True)
    source_metadata = deepcopy(source.metadata.to_dict())

    result = filter_detections_by_morphology(
        source,
        hard_area_threshold_um2=60.0,
        conditional_area_threshold_um2=48.0,
        conditions=[
            DetectionFilterCondition(
                feature="eccentricity",
                operator=">=",
                threshold=0.965,
            )
        ],
        source_name="subject_1_pointcloud.parquet",
    )

    assert result.accepted.points["detection_id"].tolist() == [11, 12]
    assert result.rejected.points["detection_id"].tolist() == [13, 14]
    assert result.rejected.points["rejection_reason"].tolist() == [
        CONDITIONAL_REJECTION_REASON,
        HARD_AREA_REJECTION_REASON,
    ]
    assert result.summary["rejected_hard_area"] == 1
    assert result.summary["rejected_conditional"] == 1
    assert result.summary["accepted_rule"] == (
        "area_um2 <= 60 and not (area_um2 > 48 and "
        "(eccentricity >= 0.965))"
    )
    parameters = result.accepted.metadata.processing.parameters
    assert parameters is not None
    assert parameters["conditions"] == [
        {"feature": "eccentricity", "operator": ">=", "threshold": 0.965}
    ]
    pd.testing.assert_frame_equal(source.points, source_points)
    assert source.metadata.to_dict() == source_metadata


def test_multiple_conditions_support_any_and_all() -> None:
    eccentricity = DetectionFilterCondition("eccentricity", ">=", 0.965)
    low_intensity = DetectionFilterCondition("mean_intensity", "<", 100.0)

    any_result = filter_detections_by_morphology(
        _dataset(),
        hard_area_threshold_um2=100.0,
        conditional_area_threshold_um2=50.0,
        conditions=[eccentricity, low_intensity],
        condition_combination="any",
    )
    all_result = filter_detections_by_morphology(
        _dataset(),
        hard_area_threshold_um2=100.0,
        conditional_area_threshold_um2=50.0,
        conditions=[eccentricity, low_intensity],
        condition_combination="all",
    )

    assert any_result.rejected.points["detection_id"].tolist() == [12, 13]
    assert all_result.rejected.points["detection_id"].tolist() == [12]


def test_conditions_are_inactive_when_not_explicitly_configured() -> None:
    result = filter_detections_by_morphology(
        _dataset(),
        hard_area_threshold_um2=100.0,
    )

    assert result.rejected.points.empty
    assert result.conditions == ()
    assert result.conditional_area_threshold_um2 is None


@pytest.mark.parametrize(
    ("conditional_area", "conditions"),
    [
        (50.0, None),
        (None, [DetectionFilterCondition("eccentricity", ">=", 0.965)]),
    ],
)
def test_conditional_area_and_conditions_must_be_paired(
    conditional_area: float | None,
    conditions: list[DetectionFilterCondition] | None,
) -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        filter_detections_by_morphology(
            _dataset(),
            hard_area_threshold_um2=100.0,
            conditional_area_threshold_um2=conditional_area,
            conditions=conditions,
        )


@pytest.mark.parametrize("conditional_area", [0.0, 60.0, 61.0, np.inf])
def test_conditional_area_must_be_positive_and_below_hard_threshold(
    conditional_area: float,
) -> None:
    with pytest.raises(ValueError, match="below hard_area_threshold_um2"):
        filter_detections_by_morphology(
            _dataset(),
            hard_area_threshold_um2=60.0,
            conditional_area_threshold_um2=conditional_area,
            conditions=[DetectionFilterCondition("eccentricity", ">=", 0.965)],
        )


def test_condition_features_must_exist_and_be_valid() -> None:
    condition = DetectionFilterCondition("eccentricity", ">=", 0.965)
    missing = _dataset(_dataset().points.drop(columns="eccentricity"))
    with pytest.raises(ValueError, match="missing condition feature"):
        filter_detections_by_morphology(
            missing,
            hard_area_threshold_um2=100.0,
            conditional_area_threshold_um2=50.0,
            conditions=[condition],
        )

    invalid_points = _dataset().points.copy()
    invalid_points.loc[1, "eccentricity"] = 1.1
    with pytest.raises(ValueError, match="values from 0 to 1"):
        filter_detections_by_morphology(
            _dataset(invalid_points),
            hard_area_threshold_um2=100.0,
            conditional_area_threshold_um2=50.0,
            conditions=[condition],
        )


def test_condition_definition_and_combination_are_validated() -> None:
    with pytest.raises(ValueError, match="condition operator"):
        DetectionFilterCondition("eccentricity", "=", 0.965)
    with pytest.raises(ValueError, match="condition threshold must be finite"):
        DetectionFilterCondition("eccentricity", ">=", np.nan)
    with pytest.raises(ValueError, match="condition_combination"):
        filter_detections_by_morphology(
            _dataset(),
            hard_area_threshold_um2=100.0,
            condition_combination="neither",  # type: ignore[arg-type]
        )


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


def test_save_pointcloud_dataset_round_trips_conditional_rejection_provenance(
    tmp_path: Path,
) -> None:
    result = filter_detections_by_morphology(
        _dataset(),
        hard_area_threshold_um2=100.0,
        conditional_area_threshold_um2=50.0,
        conditions=[DetectionFilterCondition("eccentricity", ">=", 0.965)],
    )
    table_path = tmp_path / "rejected.parquet"
    metadata_path = tmp_path / "rejected_space.json"

    paths = save_pointcloud_dataset(
        result.rejected,
        table_path,
        metadata_path,
    )
    loaded = PointCloudDataset.from_files(
        paths.table_path,
        paths.metadata_path,
        subject_name="subject_1",
    )

    pd.testing.assert_frame_equal(loaded.points, result.rejected.points)
    assert loaded.metadata.to_dict() == result.rejected.metadata.to_dict()
    assert loaded.metadata.processing is not None
    assert loaded.metadata.processing.parameters is not None
    assert loaded.metadata.processing.parameters["conditions"] == [
        {"feature": "eccentricity", "operator": ">=", "threshold": 0.965}
    ]
