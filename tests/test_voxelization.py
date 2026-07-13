import numpy as np
import pandas as pd
import tifffile

from spatialsignal.models import (
    DataRepresentation,
    DatasetMetadata,
    PointCloudDataset,
    SpaceDefinition,
)
from spatialsignal.voxelization import (
    build_nifti_ras_affine,
    build_fraction_map_from_counts,
    compute_native_voxel_denominator_grid,
    make_subject_analysis_space,
    pointcloud_to_zero_based_xyz,
    voxelize_signal_masks_to_fraction_map,
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
        schema_name="spatialsignal.dataset_metadata",
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
        schema_name="spatialsignal.dataset_metadata",
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
        schema_name="spatialsignal.dataset_metadata",
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
            [0.02, 0.0, 0.0, 0.0],
            [0.0, -0.025, 0.0, 0.0],
            [0.0, 0.0, -0.03, 0.0],
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
            [0.0, 0.0, 0.03, 0.0],
            [-0.02, 0.0, 0.0, 0.0],
            [0.0, -0.025, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_array_equal(affine, expected)


def test_compute_native_voxel_denominator_grid_uniform_case() -> None:
    source_space = SpaceDefinition(
        space_name="native_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[4, 4, 1],
        resolution_um=[1.0, 1.0, 1.0],
    )
    target_space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 2, 1],
        resolution_um=[2.0, 2.0, 1.0],
    )

    denominator = compute_native_voxel_denominator_grid(source_space, target_space)

    expected = np.full((2, 2, 1), 4, dtype=np.uint32)
    np.testing.assert_array_equal(denominator, expected)


def test_compute_native_voxel_denominator_grid_edge_case() -> None:
    source_space = SpaceDefinition(
        space_name="native_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[5, 5, 1],
        resolution_um=[1.0, 1.0, 1.0],
    )
    target_space = make_subject_analysis_space(
        source_space,
        analysis_resolution_um=[2.0, 2.0, 1.0],
    )

    denominator = compute_native_voxel_denominator_grid(source_space, target_space)

    expected = np.array(
        [
            [[4], [4], [2]],
            [[4], [4], [2]],
            [[2], [2], [1]],
        ],
        dtype=np.uint32,
    )
    np.testing.assert_array_equal(denominator, expected)


def test_build_fraction_map_from_counts() -> None:
    signal_counts = np.array(
        [
            [[3], [2]],
            [[2], [3]],
        ],
        dtype=np.uint32,
    )
    denominators = np.full((2, 2, 1), 4, dtype=np.uint32)

    fraction_map = build_fraction_map_from_counts(signal_counts, denominators)

    expected = np.array(
        [
            [[0.75], [0.5]],
            [[0.5], [0.75]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(fraction_map, expected)


def test_voxelize_signal_masks_to_fraction_map_toy_case(tmp_path) -> None:
    slice0 = np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    mask_path = tmp_path / "mask_0000.tif"
    tifffile.imwrite(mask_path, slice0)

    source_space = SpaceDefinition(
        space_name="native_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[4, 4, 1],
        resolution_um=[1.0, 1.0, 1.0],
    )
    target_space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 2, 1],
        resolution_um=[2.0, 2.0, 1.0],
    )

    voxel_map = voxelize_signal_masks_to_fraction_map(
        [mask_path],
        source_space,
        target_space,
        subject_name="Example_Subject",
    )

    expected = np.array(
        [
            [[0.75], [0.5]],
            [[0.5], [0.75]],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(voxel_map.data, expected)
    assert voxel_map.metadata.representation.kind == "voxel_map"
    assert voxel_map.metadata.representation.representation_type == "fraction_map"
    assert voxel_map.metadata.processing is not None
    assert voxel_map.metadata.processing.summary["input_signal_points"] == 10
    assert voxel_map.metadata.processing.summary["sum_signal_counts"] == 10


def test_voxelize_signal_masks_to_fraction_map_empty_signal(tmp_path) -> None:
    empty_slice = np.zeros((4, 4), dtype=np.uint8)
    mask_path = tmp_path / "mask_0000.tif"
    tifffile.imwrite(mask_path, empty_slice)

    source_space = SpaceDefinition(
        space_name="native_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[4, 4, 1],
        resolution_um=[1.0, 1.0, 1.0],
    )
    target_space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 2, 1],
        resolution_um=[2.0, 2.0, 1.0],
    )

    voxel_map = voxelize_signal_masks_to_fraction_map(
        [mask_path],
        source_space,
        target_space,
        subject_name="Example_Subject",
    )

    expected = np.zeros((2, 2, 1), dtype=np.float32)
    np.testing.assert_array_equal(voxel_map.data, expected)
    assert not np.isnan(voxel_map.data).any()
    assert voxel_map.metadata.processing is not None
    assert voxel_map.metadata.processing.summary["input_signal_points"] == 0
    assert voxel_map.metadata.processing.summary["sum_signal_counts"] == 0


def test_voxelize_signal_masks_to_fraction_map_preserves_signal_count_in_numerator(
    tmp_path,
) -> None:
    slice0 = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    slice1 = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask_paths = [tmp_path / "mask_0000.tif", tmp_path / "mask_0001.tif"]
    tifffile.imwrite(mask_paths[0], slice0)
    tifffile.imwrite(mask_paths[1], slice1)

    source_space = SpaceDefinition(
        space_name="native_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="one_based",
        units="voxel",
        shape=[4, 4, 2],
        resolution_um=[1.0, 1.0, 1.0],
    )
    target_space = SpaceDefinition(
        space_name="subject_analysis_space",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[2, 2, 2],
        resolution_um=[2.0, 2.0, 1.0],
    )

    voxel_map = voxelize_signal_masks_to_fraction_map(
        mask_paths,
        source_space,
        target_space,
        subject_name="Example_Subject",
    )

    total_signal_points = int(slice0.sum() + slice1.sum())
    denominator = compute_native_voxel_denominator_grid(source_space, target_space)
    recovered_numerator = voxel_map.data * denominator.astype(np.float32)

    assert voxel_map.metadata.processing is not None
    assert voxel_map.metadata.processing.summary["input_signal_points"] == total_signal_points
    assert voxel_map.metadata.processing.summary["sum_signal_counts"] == total_signal_points
    assert int(np.rint(recovered_numerator.sum())) == total_signal_points
