import numpy as np
import pandas as pd
import pytest
import tifffile

from spatialsignal.pointcloud import (
    CLEANED_OBJECT_REQUIRED_COLUMNS,
    SpaceDefinition,
    EDGE_COLUMNS,
    aggregate_cleaned_objects,
    build_object_membership_table,
    deduplicate_across_planes,
    summarize_deduplication_result,
)
from spatialsignal.qc import (
    duplicate_status_colormap,
    read_mask_crop,
    render_duplicate_status_mask,
    render_duplicate_status_labels,
    select_qc_plane_pairs,
)


def test_deduplicate_across_planes_builds_expected_edges_membership_and_objects() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 7, "x": 100, "y": 200, "z": 10, "x_float": 100.0, "y_float": 200.0, "area_px": 20, "major_axis_length_px": 6.0, "minor_axis_length_px": 4.0, "eccentricity": 0.4},
            {"detection_id": 2, "seg_num": 3, "x": 102, "y": 201, "z": 11, "x_float": 102.0, "y_float": 201.0, "area_px": 25, "major_axis_length_px": 8.0, "minor_axis_length_px": 5.0, "eccentricity": 0.6},
            {"detection_id": 3, "seg_num": 5, "x": 101, "y": 199, "z": 12, "x_float": 101.0, "y_float": 199.0, "area_px": 22, "major_axis_length_px": 7.0, "minor_axis_length_px": 4.5, "eccentricity": 0.5},
            {"detection_id": 4, "seg_num": 8, "x": 300, "y": 400, "z": 11, "x_float": 300.0, "y_float": 400.0, "area_px": 18, "major_axis_length_px": 5.0, "minor_axis_length_px": 3.0, "eccentricity": 0.3},
            {"detection_id": 5, "seg_num": 2, "x": 303, "y": 402, "z": 12, "x_float": 303.0, "y_float": 402.0, "area_px": 21, "major_axis_length_px": 7.0, "minor_axis_length_px": 4.0, "eccentricity": 0.5},
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
                "mean_major_axis_length_px": 7.0,
                "mean_minor_axis_length_px": 4.5,
                "mean_eccentricity": 0.5,
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
                "mean_major_axis_length_px": 6.0,
                "mean_minor_axis_length_px": 3.5,
                "mean_eccentricity": 0.4,
            },
        ],
        columns=CLEANED_OBJECT_REQUIRED_COLUMNS
        + [
            "mean_area_px",
            "max_area_px",
            "mean_major_axis_length_px",
            "mean_minor_axis_length_px",
            "mean_eccentricity",
        ],
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


def test_aggregate_cleaned_objects_sorts_unsorted_membership() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 10, "y": 20, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 12, "y": 22, "z": 2},
            {"detection_id": 3, "seg_num": 3, "x": 30, "y": 40, "z": 3},
        ]
    )
    membership = pd.DataFrame(
        [
            {"object_id": 2, "detection_id": 3},
            {"object_id": 1, "detection_id": 2},
            {"object_id": 1, "detection_id": 1},
        ]
    )

    objects = aggregate_cleaned_objects(points, membership)

    assert list(objects["object_id"]) == [1, 2]
    assert list(objects["x_float"]) == [11.0, 30.0]
    assert list(objects["n_detections"]) == [2, 1]


def test_aggregate_cleaned_objects_rejects_unknown_detection_ids() -> None:
    points = pd.DataFrame(
        [{"detection_id": 1, "seg_num": 1, "x": 10, "y": 20, "z": 1}]
    )
    membership = pd.DataFrame([{"object_id": 1, "detection_id": 999}])

    with pytest.raises(ValueError, match="absent from the point cloud"):
        aggregate_cleaned_objects(points, membership)


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


def test_deduplicate_across_planes_rejects_indirect_same_plane_merge() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 100, "y": 100, "z": 10},
            {"detection_id": 2, "seg_num": 2, "x": 103, "y": 100, "z": 10},
            {"detection_id": 3, "seg_num": 3, "x": 101, "y": 100, "z": 11},
        ]
    )
    space = SpaceDefinition(
        space_name="subject_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[500, 500, 20],
        resolution_um=[1.0, 1.0, 5.0],
    )

    result = deduplicate_across_planes(
        points,
        space,
        max_plane_offset=1,
        max_xy_distance_um=3.0,
        max_n_planes=2,
    )

    expected_membership = pd.DataFrame(
        [
            {"object_id": 1, "detection_id": 1},
            {"object_id": 1, "detection_id": 3},
            {"object_id": 2, "detection_id": 2},
        ]
    )
    pd.testing.assert_frame_equal(result.membership, expected_membership)
    assert len(result.edges) == 1
    assert (result.objects["n_detections"] == result.objects["n_planes"]).all()


def test_component_merge_rejects_plane_overlap_created_by_separate_chains() -> None:
    points = pd.DataFrame(
        {
            "detection_id": [1, 2, 3, 4],
            "z": [10, 11, 10, 12],
        }
    )
    edges = pd.DataFrame(
        [
            {
                "source_detection_id": 1,
                "target_detection_id": 2,
                "source_z": 10,
                "target_z": 11,
                "plane_offset": 1,
                "dx_um": 0.1,
                "dy_um": 0.0,
                "xy_distance_um": 0.1,
                "dz_um": 5.0,
            },
            {
                "source_detection_id": 3,
                "target_detection_id": 4,
                "source_z": 10,
                "target_z": 12,
                "plane_offset": 2,
                "dx_um": 0.2,
                "dy_um": 0.0,
                "xy_distance_um": 0.2,
                "dz_um": 10.0,
            },
            {
                "source_detection_id": 2,
                "target_detection_id": 4,
                "source_z": 11,
                "target_z": 12,
                "plane_offset": 1,
                "dx_um": 0.3,
                "dy_um": 0.0,
                "xy_distance_um": 0.3,
                "dz_um": 5.0,
            },
        ],
        columns=EDGE_COLUMNS,
    )

    membership, accepted_edges = build_object_membership_table(
        points,
        edges,
        max_n_planes=3,
    )

    expected_membership = pd.DataFrame(
        [
            {"object_id": 1, "detection_id": 1},
            {"object_id": 1, "detection_id": 2},
            {"object_id": 2, "detection_id": 3},
            {"object_id": 2, "detection_id": 4},
        ]
    )
    pd.testing.assert_frame_equal(membership, expected_membership)
    assert len(accepted_edges) == 2


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


def test_read_and_render_duplicate_mask_crop(tmp_path) -> None:
    mask = np.zeros((8, 10), dtype=np.uint16)
    mask[2:5, 3:6] = 7
    mask[4:7, 6:9] = 9
    mask_path = tmp_path / "masks_plane_0001.tif"
    tifffile.imwrite(mask_path, mask)

    crop = read_mask_crop(
        mask_path,
        x_start=2,
        x_end=9,
        y_start=1,
        y_end=7,
    )
    rendered = render_duplicate_status_mask(crop, {7})
    status = render_duplicate_status_labels(crop, {7})
    colormap = duplicate_status_colormap()

    assert crop.shape == (6, 7)
    assert tuple(rendered[2, 2]) == (0, 255, 0)
    assert tuple(rendered[4, 5]) == (150, 25, 25)
    assert tuple(rendered[0, 0]) == (0, 0, 0)
    assert status[2, 2] == 2
    assert status[4, 5] == 1
    assert status[0, 0] == 0
    assert tuple(colormap[:, 1]) == (65535, 0, 0)
    assert tuple(colormap[:, 2]) == (0, 65535, 0)
