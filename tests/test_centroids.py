from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile

from lsfm_cell_mapping.pointcloud import (
    DataRepresentation,
    DatasetMetadata,
    POINTCLOUD_COLUMNS,
    POINTCLOUD_REQUIRED_COLUMNS,
    PointCloudDataset,
    ProcessingProvenance,
    SpaceDefinition,
    build_pointcloud_from_masks,
    extract_centroids_from_mask,
    extract_centroids_from_mask_stack,
    matlab_round,
)
from lsfm_cell_mapping.qc import pointcloud_slice_to_image


def test_matlab_round_matches_expected_half_up_behavior() -> None:
    assert matlab_round(10.5) == 11
    assert matlab_round(11.5) == 12
    assert matlab_round(10.49) == 10


def test_extract_centroids_from_mask_returns_expected_columns_and_values(tmp_path: Path) -> None:
    mask = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [2, 2, 0],
        ],
        dtype=np.uint16,
    )
    mask_path = tmp_path / "masks_1.tif"
    tifffile.imwrite(mask_path, mask)

    pointcloud = extract_centroids_from_mask(mask_path, slice_index=1)

    expected = pd.DataFrame(
        [
            {
                "detection_id": 1,
                "seg_num": 1,
                "x": 3,
                "y": 1,
                "z": 1,
                "area_px": 2,
                "x_float": 2.5,
                "y_float": 1.0,
                "major_axis_length_px": 2.0,
                "minor_axis_length_px": 0.0,
                "eccentricity": 1.0,
            },
            {
                "detection_id": 2,
                "seg_num": 2,
                "x": 2,
                "y": 3,
                "z": 1,
                "area_px": 2,
                "x_float": 1.5,
                "y_float": 3.0,
                "major_axis_length_px": 2.0,
                "minor_axis_length_px": 0.0,
                "eccentricity": 1.0,
            },
        ],
        columns=POINTCLOUD_COLUMNS,
    )

    pd.testing.assert_frame_equal(pointcloud.reset_index(drop=True), expected)


def test_extract_centroids_from_mask_returns_empty_table_for_empty_mask(
    tmp_path: Path,
) -> None:
    mask = np.zeros((3, 3), dtype=np.uint16)
    mask_path = tmp_path / "masks_1.tif"
    tifffile.imwrite(mask_path, mask)

    pointcloud = extract_centroids_from_mask(mask_path, slice_index=1)

    assert list(pointcloud.columns) == POINTCLOUD_COLUMNS
    assert pointcloud.empty


def test_extract_centroids_from_mask_stack_uses_slice_order(tmp_path: Path) -> None:
    mask1 = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [2, 2, 0],
        ],
        dtype=np.uint16,
    )
    mask2 = np.array(
        [
            [0, 0, 3],
            [0, 3, 3],
            [0, 0, 0],
        ],
        dtype=np.uint16,
    )

    path2 = tmp_path / "masks_2.tif"
    path1 = tmp_path / "masks_1.tif"
    tifffile.imwrite(path2, mask2)
    tifffile.imwrite(path1, mask1)

    pointcloud = extract_centroids_from_mask_stack([path1, path2], max_workers=1)

    expected_xyz = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 3, "y": 1, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 2, "y": 3, "z": 1},
            {"detection_id": 3, "seg_num": 3, "x": 3, "y": 2, "z": 2},
        ]
    )

    pd.testing.assert_frame_equal(
        pointcloud[POINTCLOUD_REQUIRED_COLUMNS].reset_index(drop=True),
        expected_xyz,
    )


def test_pointcloud_slice_to_image_marks_expected_pixels() -> None:
    slice_points = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 3, "y": 1, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 2, "y": 3, "z": 1},
        ]
    )

    image = pointcloud_slice_to_image(slice_points, (3, 3), one_based=True)

    expected = np.array(
        [
            [0, 0, 255],
            [0, 0, 0],
            [0, 255, 0],
        ],
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(image, expected)


def test_build_pointcloud_from_masks_writes_csv_and_space_json(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    mask_dir.mkdir()

    mask = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [2, 2, 0],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(mask_dir / "masks_1.tif", mask)

    build_pointcloud_from_masks(
        mask_dir=mask_dir,
        out_dir=out_dir,
        subject_name="Test Subject",
        space_name="subject_space",
        orientation="las",
        resolution_um=[1.8, 1.8, 5.0],
        max_workers=1,
    )

    assert (out_dir / "Test_Subject_pointcloud.csv").exists()
    assert (out_dir / "Test_Subject_pointcloud_space.json").exists()
    pointcloud = pd.read_csv(out_dir / "Test_Subject_pointcloud.csv")
    assert list(pointcloud["detection_id"]) == [1, 2]
    with (out_dir / "Test_Subject_pointcloud_space.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["representation"]["kind"] == "point_cloud"
    assert metadata["representation"]["representation_type"] == "point_centroids"


def test_dataset_metadata_round_trip_json(tmp_path: Path) -> None:
    metadata = DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[10, 20, 30],
            resolution_um=[1.8, 1.8, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
        processing=ProcessingProvenance(
            stage="deduplicate_across_planes",
            parameters={"max_plane_offset": 1, "max_xy_distance_um": 3.0, "max_n_planes": 2},
            summary={"raw_detections": 10, "cleaned_objects": 8, "accepted_edges": 2},
        ),
    )

    json_path = tmp_path / "space.json"
    metadata.to_json(json_path)
    loaded = DatasetMetadata.from_json(json_path)

    assert loaded == metadata


def test_pointcloud_dataset_from_files_and_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "Test_Subject_pointcloud.csv"
    json_path = tmp_path / "Test_Subject_pointcloud_space.json"

    pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 3, "y": 1, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 2, "y": 3, "z": 1},
        ],
        columns=POINTCLOUD_REQUIRED_COLUMNS,
    ).to_csv(csv_path, index=False)

    DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[3, 3, 1],
            resolution_um=[1.8, 1.8, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    ).to_json(json_path)

    dataset = PointCloudDataset.from_files(csv_path, json_path)
    dataset.validate()
    summary = dataset.summary()

    assert dataset.subject_name == "Test_Subject"
    assert summary["n_points"] == 2
    assert summary["space_name"] == "subject_space"
    assert summary["x_min"] == 2
    assert summary["y_max"] == 3
    assert dataset.metadata.representation.kind == "point_cloud"


def test_pointcloud_dataset_validate_raises_for_out_of_bounds_points(tmp_path: Path) -> None:
    csv_path = tmp_path / "Test_Subject_pointcloud.csv"
    json_path = tmp_path / "Test_Subject_pointcloud_space.json"

    pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 4, "y": 1, "z": 1},
        ],
        columns=POINTCLOUD_REQUIRED_COLUMNS,
    ).to_csv(csv_path, index=False)

    DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[3, 3, 1],
            resolution_um=[1.8, 1.8, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    ).to_json(json_path)

    dataset = PointCloudDataset.from_files(csv_path, json_path)

    try:
        dataset.validate()
    except ValueError as exc:
        assert "x values above" in str(exc)
    else:
        raise AssertionError("Expected dataset.validate() to raise for out-of-bounds points")


def _make_valid_dataset_files(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "Test_Subject_pointcloud.csv"
    json_path = tmp_path / "Test_Subject_pointcloud_space.json"

    pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 3, "y": 1, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 2, "y": 3, "z": 1},
        ],
        columns=POINTCLOUD_REQUIRED_COLUMNS,
    ).to_csv(csv_path, index=False)

    DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="one_based",
            units="voxel",
            shape=[3, 3, 1],
            resolution_um=[1.8, 1.8, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    ).to_json(json_path)

    return csv_path, json_path


def test_dataset_columns_invalid(tmp_path: Path) -> None:
    csv_path, json_path = _make_valid_dataset_files(tmp_path)

    pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 3, "y": 1},
        ]
    ).to_csv(csv_path, index=False)

    dataset = PointCloudDataset.from_files(csv_path, json_path)

    try:
        dataset.validate()
    except ValueError as exc:
        assert "required columns" in str(exc)
    else:
        raise AssertionError("Expected dataset.validate() to raise for missing required columns")


def test_dataset_axis_metadata_invalid(tmp_path: Path) -> None:
    csv_path, json_path = _make_valid_dataset_files(tmp_path)

    DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            axis_labels=["x", "y"],
            indexing="one_based",
            units="voxel",
            shape=[3, 3, 1],
            resolution_um=[1.8, 1.8, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    ).to_json(json_path)

    dataset = PointCloudDataset.from_files(csv_path, json_path)

    try:
        dataset.validate()
    except ValueError as exc:
        assert "axis_labels" in str(exc)
    else:
        raise AssertionError("Expected dataset.validate() to raise for invalid axis metadata")


def test_dataset_indexing_invalid(tmp_path: Path) -> None:
    csv_path, json_path = _make_valid_dataset_files(tmp_path)

    DatasetMetadata(
        schema_name="lsfm_cell_mapping.dataset_metadata",
        schema_version="0.1.0",
        space=SpaceDefinition(
            space_name="subject_space",
            orientation="las",
            axis_labels=["x", "y", "z"],
            indexing="bad_mode",
            units="voxel",
            shape=[3, 3, 1],
            resolution_um=[1.8, 1.8, 5.0],
        ),
        representation=DataRepresentation(
            kind="point_cloud",
            representation_type="point_centroids",
        ),
    ).to_json(json_path)

    dataset = PointCloudDataset.from_files(csv_path, json_path)

    try:
        dataset.validate()
    except ValueError as exc:
        assert "indexing must be one of" in str(exc)
    else:
        raise AssertionError("Expected dataset.validate() to raise for invalid indexing")


def test_dataset_bounds_invalid(tmp_path: Path) -> None:
    csv_path, json_path = _make_valid_dataset_files(tmp_path)

    pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 4, "y": 1, "z": 1},
        ],
        columns=POINTCLOUD_REQUIRED_COLUMNS,
    ).to_csv(csv_path, index=False)

    dataset = PointCloudDataset.from_files(csv_path, json_path)

    try:
        dataset.validate()
    except ValueError as exc:
        assert "x values above" in str(exc)
    else:
        raise AssertionError("Expected dataset.validate() to raise for out-of-bounds coordinates")
