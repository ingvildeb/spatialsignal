import pandas as pd
import pytest

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.pointcloud import (
    MATCH_COLUMNS,
    map_detection_matches_to_objects,
    match_colocalized_detections,
)


def _dataset(
    subject_name: str,
    rows: list[dict[str, int | float]],
    *,
    shape: list[int] | None = None,
    resolution_um: list[float] | None = None,
    indexing: str = "zero_based",
) -> PointCloudDataset:
    shape = shape or [100, 100, 3]
    resolution_um = resolution_um or [1.0, 1.0, 5.0]
    points = pd.DataFrame(
        rows,
        columns=["detection_id", "seg_num", "x", "y", "z", "x_float", "y_float"],
    )
    metadata = DatasetMetadata(
        schema_name="spatialsignal.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing=indexing,
            units="voxel",
            shape=shape,
            resolution_um=resolution_um,
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    return PointCloudDataset(
        subject_name=subject_name,
        points=points,
        metadata=metadata,
    )


def _point(detection_id: int, x: float, y: float, z: int) -> dict[str, int | float]:
    return {
        "detection_id": detection_id,
        "seg_num": detection_id,
        "x": int(x),
        "y": int(y),
        "z": z,
        "x_float": x,
        "y_float": y,
    }


def test_matching_is_exact_z_and_reports_every_declared_plane() -> None:
    dataset_a = _dataset("subject", [_point(1, 10, 10, 0), _point(2, 20, 20, 1)])
    dataset_b = _dataset("subject", [_point(11, 11, 10, 0), _point(12, 20, 20, 2)])

    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=2.0,
    )

    assert list(result.matches["a_detection_id"]) == [1]
    assert list(result.matches["b_detection_id"]) == [11]
    assert list(result.matches["z"]) == [0]
    assert list(result.plane_summary["z"]) == [0, 1, 2]
    assert list(result.plane_summary["n_accepted_matches"]) == [1, 0, 0]


def test_matching_maximizes_cardinality_before_minimizing_distance() -> None:
    dataset_a = _dataset("subject", [_point(1, 0, 10, 0), _point(2, 2, 10, 0)])
    dataset_b = _dataset("subject", [_point(11, 1, 10, 0), _point(12, 0, 10, 0)])

    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=1.1,
    )

    assert set(
        result.matches[["a_detection_id", "b_detection_id"]].itertuples(
            index=False,
            name=None,
        )
    ) == {(1, 12), (2, 11)}
    plane = result.plane_summary.loc[result.plane_summary["z"] == 0].iloc[0]
    assert plane["n_candidate_pairs"] == 3
    assert plane["n_ambiguous_a"] == 1
    assert plane["n_ambiguous_b"] == 1
    assert result.matches["is_ambiguous"].all()


def test_matching_uses_xy_resolution_in_microns() -> None:
    dataset_a = _dataset(
        "subject",
        [_point(1, 10, 10, 0)],
        resolution_um=[2.0, 1.0, 5.0],
    )
    dataset_b = _dataset(
        "subject",
        [_point(11, 12, 10, 0)],
        resolution_um=[2.0, 1.0, 5.0],
    )

    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=3.0,
    )

    assert result.matches.empty


def test_matching_handles_empty_channel_and_one_based_planes() -> None:
    dataset_a = _dataset(
        "subject",
        [],
        shape=[20, 20, 2],
        indexing="one_based",
    )
    dataset_b = _dataset(
        "subject",
        [_point(11, 2, 2, 1)],
        shape=[20, 20, 2],
        indexing="one_based",
    )

    result = match_colocalized_detections(
        dataset_a,
        dataset_b,
        max_xy_distance_um=2.0,
    )

    assert result.matches.empty
    assert list(result.plane_summary["z"]) == [1, 2]
    assert result.plane_summary.loc[0, "n_unmatched_b"] == 1


def test_matching_rejects_incompatible_spaces() -> None:
    dataset_a = _dataset("subject", [_point(1, 10, 10, 0)])
    dataset_b = _dataset(
        "subject",
        [_point(11, 10, 10, 0)],
        resolution_um=[2.0, 1.0, 5.0],
    )

    with pytest.raises(ValueError, match="resolution_um"):
        match_colocalized_detections(
            dataset_a,
            dataset_b,
            max_xy_distance_um=3.0,
        )


def test_object_mapping_preserves_cross_channel_deduplication_conflict() -> None:
    matches = pd.DataFrame(
        [
            {
                "a_detection_id": 1,
                "b_detection_id": 11,
                "z": 0,
                "dx_um": 1.0,
                "dy_um": 0.0,
                "xy_distance_um": 1.0,
                "a_candidate_count": 1,
                "b_candidate_count": 1,
                "is_ambiguous": False,
            },
            {
                "a_detection_id": 2,
                "b_detection_id": 12,
                "z": 1,
                "dx_um": 1.5,
                "dy_um": 0.0,
                "xy_distance_um": 1.5,
                "a_candidate_count": 1,
                "b_candidate_count": 1,
                "is_ambiguous": False,
            },
        ],
        columns=MATCH_COLUMNS,
    )
    a_membership = pd.DataFrame(
        [
            {"object_id": 1, "detection_id": 1},
            {"object_id": 1, "detection_id": 2},
        ]
    )
    b_membership = pd.DataFrame(
        [
            {"object_id": 10, "detection_id": 11},
            {"object_id": 20, "detection_id": 12},
        ]
    )

    result = map_detection_matches_to_objects(matches, a_membership, b_membership)

    assert len(result.relationships) == 2
    assert set(result.relationships["relationship_status"]) == {
        "one_a_to_multiple_b"
    }
    assert set(result.relationships["a_partner_count"]) == {2}
    assert set(result.relationships["b_partner_count"]) == {1}


def test_object_mapping_requires_membership_for_every_detection() -> None:
    matches = pd.DataFrame(
        [
            {
                "a_detection_id": 1,
                "b_detection_id": 11,
                "z": 0,
                "dx_um": 0.0,
                "dy_um": 0.0,
                "xy_distance_um": 0.0,
                "a_candidate_count": 1,
                "b_candidate_count": 1,
                "is_ambiguous": False,
            }
        ],
        columns=MATCH_COLUMNS,
    )

    with pytest.raises(ValueError, match="Every matched detection"):
        map_detection_matches_to_objects(
            matches,
            pd.DataFrame([{"object_id": 1, "detection_id": 999}]),
            pd.DataFrame([{"object_id": 2, "detection_id": 11}]),
        )
