import numpy as np
import pandas as pd

from lsfm_cell_mapping.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from lsfm_cell_mapping.voxelization import (
    build_nifti_ras_affine,
    make_subject_analysis_space,
    pointcloud_to_zero_based_xyz,
    voxelize_point_centroids_to_count_map,
    voxelize_to_space,
)


def _make_centroid_dataset() -> PointCloudDataset:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 1, "y": 1, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 2, "y": 2, "z": 2},
            {"detection_id": 3, "seg_num": 3, "x": 4, "y": 4, "z": 4},
        ]
    )
    metadata = DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[4, 4, 4],
            resolution_um=[5.0, 5.0, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    return PointCloudDataset(
        subject_name="Test_Subject",
        points=points,
        metadata=metadata,
    )


def _make_target_space() -> SpaceDefinition:
    return SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 2, 2],
        resolution_um=[10.0, 10.0, 10.0],
    )


def test_make_subject_analysis_space_inherits_native_spatial_conventions() -> None:
    dataset = _make_centroid_dataset()

    analysis_space = make_subject_analysis_space(
        dataset.space,
        analysis_resolution_um=[10.0, 10.0, 10.0],
    )

    assert analysis_space.space_name == "subject_analysis_space"
    assert analysis_space.orientation == dataset.space.orientation
    assert analysis_space.axis_labels == dataset.space.axis_labels
    assert analysis_space.indexing == dataset.space.indexing
    assert analysis_space.units == dataset.space.units
    assert analysis_space.shape == [2, 2, 2]
    assert analysis_space.resolution_um == [10.0, 10.0, 10.0]


def test_pointcloud_to_zero_based_xyz_converts_one_based_coordinates() -> None:
    dataset = _make_centroid_dataset()

    xyz = pointcloud_to_zero_based_xyz(dataset)

    expected = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(xyz, expected)


def test_voxelize_point_centroids_to_count_map_accumulates_counts() -> None:
    dataset = _make_centroid_dataset()
    target_space = _make_target_space()

    voxel_map = voxelize_point_centroids_to_count_map(dataset, target_space)

    expected = np.zeros((2, 2, 2), dtype=np.uint32)
    expected[0, 0, 0] = 2
    expected[1, 1, 1] = 1
    np.testing.assert_array_equal(voxel_map.data, expected)
    assert voxel_map.metadata.representation.kind == "voxel_map"
    assert voxel_map.metadata.representation.representation_type == "count_map"
    assert voxel_map.metadata.processing is not None
    assert voxel_map.metadata.processing.summary["total_count"] == 3


def test_voxelize_to_space_dispatches_for_point_centroids() -> None:
    dataset = _make_centroid_dataset()
    target_space = _make_target_space()

    voxel_map = voxelize_to_space(dataset, target_space)

    assert voxel_map.subject_name == "Test_Subject"
    assert voxel_map.metadata.representation.kind == "voxel_map"
    assert voxel_map.metadata.representation.representation_type == "count_map"
    assert voxel_map.metadata.processing is not None
    assert voxel_map.metadata.processing.parameters["source_representation_type"] == "point_centroids"


def test_voxelize_to_space_allows_multiple_points_same_target_voxel() -> None:
    points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 1, "y": 1, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 2, "y": 2, "z": 2},
        ]
    )
    metadata = DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[2, 2, 2],
            resolution_um=[5.0, 5.0, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    dataset = PointCloudDataset(
        subject_name="Test_Subject",
        points=points,
        metadata=metadata,
    )
    target_space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[1, 1, 1],
        resolution_um=[10.0, 10.0, 10.0],
    )

    voxel_map = voxelize_to_space(dataset, target_space)

    expected = np.array([[[2]]], dtype=np.uint32)
    np.testing.assert_array_equal(voxel_map.data, expected)


def test_make_subject_analysis_space_uses_ceiling_to_preserve_full_extent() -> None:
    source_space = SpaceDefinition(
        space_name="native_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[5, 5, 5],
        resolution_um=[3.0, 3.0, 3.0],
    )

    analysis_space = make_subject_analysis_space(
        source_space,
        analysis_resolution_um=[4.0, 4.0, 4.0],
    )

    assert analysis_space.shape == [4, 4, 4]


def test_cleaned_object_table_can_follow_shared_voxelization_path() -> None:
    points = pd.DataFrame(
        [
            {"object_id": 1, "x": 1, "y": 1, "z": 1, "n_detections": 1},
            {"object_id": 2, "x": 2, "y": 2, "z": 2, "n_detections": 2},
        ]
    )
    metadata = DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="native_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[2, 2, 2],
            resolution_um=[5.0, 5.0, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    )
    dataset = PointCloudDataset(
        subject_name="Test_Subject",
        points=points,
        metadata=metadata,
    )
    dataset.validate_spatial_points()

    target_space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[1, 1, 1],
        resolution_um=[10.0, 10.0, 10.0],
    )

    voxel_map = voxelize_to_space(dataset, target_space)

    expected = np.array([[[2]]], dtype=np.uint32)
    np.testing.assert_array_equal(voxel_map.data, expected)


def test_build_nifti_ras_affine_converts_brainglobe_origin_convention() -> None:
    space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[2, 2, 2],
        resolution_um=[20.0, 25.0, 30.0],
    )

    affine = build_nifti_ras_affine(space)

    expected = np.array(
        [
            [20.0, 0.0, 0.0, 0.0],
            [0.0, -25.0, 0.0, 0.0],
            [0.0, 0.0, -30.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(affine, expected)


def test_build_nifti_ras_affine_handles_nontrivial_axis_world_mapping() -> None:
    space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="asl",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[2, 2, 2],
        resolution_um=[20.0, 25.0, 30.0],
    )

    affine = build_nifti_ras_affine(space)

    expected = np.array(
        [
            [0.0, 0.0, 30.0, 0.0],
            [-20.0, 0.0, 0.0, 0.0],
            [0.0, -25.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(affine, expected)
