import pandas as pd

from lsfm_cell_mapping.pointcloud import (
    CLEANED_OBJECT_REQUIRED_COLUMNS,
    SpaceDefinition,
    EDGE_COLUMNS,
    deduplicate_across_planes,
    summarize_deduplication_result,
)
from lsfm_cell_mapping.qc import select_qc_plane_pairs


def test_deduplicate_across_planes_builds_expected_edges_membership_and_objects() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 7, "x": 100, "y": 200, "z": 10, "x_float": 100.0, "y_float": 200.0, "area_px": 20},
            {"detection_id": 2, "seg_num": 3, "x": 102, "y": 201, "z": 11, "x_float": 102.0, "y_float": 201.0, "area_px": 25},
            {"detection_id": 3, "seg_num": 5, "x": 101, "y": 199, "z": 12, "x_float": 101.0, "y_float": 199.0, "area_px": 22},
            {"detection_id": 4, "seg_num": 8, "x": 300, "y": 400, "z": 11, "x_float": 300.0, "y_float": 400.0, "area_px": 18},
            {"detection_id": 5, "seg_num": 2, "x": 303, "y": 402, "z": 12, "x_float": 303.0, "y_float": 402.0, "area_px": 21},
        ]
    )

    space = SpaceDefinition(
        space_name="subject_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[500, 500, 20],
        resolution_um=[1.0, 1.0, 5.0],
    )

    result = deduplicate_across_planes(
        points,
        space,
        max_plane_offset=1,
        max_xy_distance_um=5.0,
    )

    expected_edges = pd.DataFrame(
        [
            {
                "source_detection_id": 1,
                "target_detection_id": 2,
                "source_z": 10,
                "target_z": 11,
                "plane_offset": 1,
                "dx_um": 2.0,
                "dy_um": 1.0,
                "xy_distance_um": 2.23606797749979,
                "dz_um": 5.0,
            },
            {
                "source_detection_id": 2,
                "target_detection_id": 3,
                "source_z": 11,
                "target_z": 12,
                "plane_offset": 1,
                "dx_um": -1.0,
                "dy_um": -2.0,
                "xy_distance_um": 2.23606797749979,
                "dz_um": 5.0,
            },
            {
                "source_detection_id": 4,
                "target_detection_id": 5,
                "source_z": 11,
                "target_z": 12,
                "plane_offset": 1,
                "dx_um": 3.0,
                "dy_um": 2.0,
                "xy_distance_um": 3.605551275463989,
                "dz_um": 5.0,
            },
        ],
        columns=EDGE_COLUMNS,
    )

    pd.testing.assert_frame_equal(result.edges.reset_index(drop=True), expected_edges)

    expected_membership = pd.DataFrame(
        [
            {"object_id": 1, "detection_id": 1},
            {"object_id": 1, "detection_id": 2},
            {"object_id": 1, "detection_id": 3},
            {"object_id": 2, "detection_id": 4},
            {"object_id": 2, "detection_id": 5},
        ]
    )
    pd.testing.assert_frame_equal(result.membership.reset_index(drop=True), expected_membership)

    expected_objects = pd.DataFrame(
        [
            {
                "object_id": 1,
                "x": 101,
                "y": 200,
                "z": 11,
                "x_float": 101.0,
                "y_float": 200.0,
                "z_float": 11.0,
                "n_detections": 3,
                "n_planes": 3,
                "z_min": 10,
                "z_max": 12,
                "mean_area_px": (20.0 + 25.0 + 22.0) / 3.0,
                "max_area_px": 25.0,
            },
            {
                "object_id": 2,
                "x": 302,
                "y": 401,
                "z": 12,
                "x_float": 301.5,
                "y_float": 401.0,
                "z_float": 11.5,
                "n_detections": 2,
                "n_planes": 2,
                "z_min": 11,
                "z_max": 12,
                "mean_area_px": 19.5,
                "max_area_px": 21.0,
            },
        ],
        columns=CLEANED_OBJECT_REQUIRED_COLUMNS + ["mean_area_px", "max_area_px"],
    )

    pd.testing.assert_frame_equal(result.objects.reset_index(drop=True), expected_objects)


def test_deduplicate_across_planes_falls_back_to_integer_coordinates_when_float_columns_missing() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 10, "y": 10, "z": 1},
            {"detection_id": 2, "seg_num": 1, "x": 11, "y": 10, "z": 2},
        ]
    )

    space = SpaceDefinition(
        space_name="subject_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[20, 20, 5],
        resolution_um=[1.0, 1.0, 5.0],
    )

    result = deduplicate_across_planes(
        points,
        space,
        max_plane_offset=1,
        max_xy_distance_um=2.0,
    )

    assert len(result.edges) == 1
    assert len(result.objects) == 1
    assert result.objects.loc[0, "x_float"] == 10.5


def test_deduplicate_across_planes_max_n_planes_cap_blocks_three_plane_chain() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 7, "x": 100, "y": 200, "z": 10, "x_float": 100.0, "y_float": 200.0},
            {"detection_id": 2, "seg_num": 3, "x": 102, "y": 201, "z": 11, "x_float": 102.0, "y_float": 201.0},
            {"detection_id": 3, "seg_num": 5, "x": 101, "y": 199, "z": 12, "x_float": 101.0, "y_float": 199.0},
        ]
    )

    space = SpaceDefinition(
        space_name="subject_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[500, 500, 20],
        resolution_um=[1.0, 1.0, 20.0],
    )

    result = deduplicate_across_planes(
        points,
        space,
        max_plane_offset=1,
        max_xy_distance_um=5.0,
        max_n_planes=2,
    )

    expected_edges = pd.DataFrame(
        [
            {
                "source_detection_id": 1,
                "target_detection_id": 2,
                "source_z": 10,
                "target_z": 11,
                "plane_offset": 1,
                "dx_um": 2.0,
                "dy_um": 1.0,
                "xy_distance_um": 2.23606797749979,
                "dz_um": 20.0,
            }
        ],
        columns=EDGE_COLUMNS,
    )
    pd.testing.assert_frame_equal(result.edges.reset_index(drop=True), expected_edges)

    expected_membership = pd.DataFrame(
        [
            {"object_id": 1, "detection_id": 1},
            {"object_id": 1, "detection_id": 2},
            {"object_id": 2, "detection_id": 3},
        ]
    )
    pd.testing.assert_frame_equal(result.membership.reset_index(drop=True), expected_membership)

    assert list(result.objects["n_planes"]) == [2, 1]


def test_summarize_deduplication_result_reports_minimal_counts() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 10, "y": 10, "z": 1},
            {"detection_id": 2, "seg_num": 1, "x": 11, "y": 10, "z": 2},
        ]
    )

    space = SpaceDefinition(
        space_name="subject_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[20, 20, 5],
        resolution_um=[1.0, 1.0, 5.0],
    )
    result = deduplicate_across_planes(
        points,
        space,
        max_plane_offset=1,
        max_xy_distance_um=2.0,
    )

    summary = summarize_deduplication_result(points, result)

    assert summary == {
        "raw_detections": 2,
        "cleaned_objects": 1,
        "accepted_edges": 1,
    }


def test_select_qc_plane_pairs_prefers_central_populated_adjacent_pairs() -> None:
    rows = []
    detection_id = 1
    for z, n in [(1, 1), (2, 1), (3, 10), (4, 15), (5, 50), (6, 60), (7, 20), (8, 10), (9, 1), (10, 1)]:
        for _ in range(n):
            rows.append({"detection_id": detection_id, "seg_num": detection_id, "x": 1, "y": 1, "z": z})
            detection_id += 1

    points = pd.DataFrame(rows)
    pairs = select_qc_plane_pairs(points, n_pairs=2)

    assert pairs == [(3, 4), (7, 8)]
