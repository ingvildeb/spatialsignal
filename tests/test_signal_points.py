from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile

from lsfm_cell_mapping.pointcloud import (
    SIGNAL_POINT_REQUIRED_COLUMNS,
    build_signal_points_from_masks,
    extract_signal_points_from_mask,
    extract_signal_points_from_mask_stack,
)


def test_extract_signal_points_from_mask_returns_expected_rows(tmp_path: Path) -> None:
    mask = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask_path = tmp_path / "signal_1.tif"
    tifffile.imwrite(mask_path, mask)

    points = extract_signal_points_from_mask(mask_path, slice_index=2)

    expected = pd.DataFrame(
        [
            {"point_id": 1, "x": 2, "y": 1, "z": 2},
            {"point_id": 2, "x": 1, "y": 2, "z": 2},
            {"point_id": 3, "x": 3, "y": 2, "z": 2},
        ],
        columns=SIGNAL_POINT_REQUIRED_COLUMNS,
    )
    pd.testing.assert_frame_equal(points.reset_index(drop=True), expected)


def test_extract_signal_points_from_mask_returns_empty_table_for_empty_mask(
    tmp_path: Path,
) -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask_path = tmp_path / "signal_1.tif"
    tifffile.imwrite(mask_path, mask)

    points = extract_signal_points_from_mask(mask_path, slice_index=1)

    assert list(points.columns) == SIGNAL_POINT_REQUIRED_COLUMNS
    assert points.empty


def test_extract_signal_points_from_mask_stack_uses_slice_order(tmp_path: Path) -> None:
    mask1 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask2 = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    path2 = tmp_path / "signal_2.tif"
    path1 = tmp_path / "signal_1.tif"
    tifffile.imwrite(path2, mask2)
    tifffile.imwrite(path1, mask1)

    points = extract_signal_points_from_mask_stack([path1, path2], max_workers=1)

    expected = pd.DataFrame(
        [
            {"point_id": 1, "x": 2, "y": 1, "z": 1},
            {"point_id": 2, "x": 1, "y": 3, "z": 1},
            {"point_id": 3, "x": 3, "y": 1, "z": 2},
            {"point_id": 4, "x": 2, "y": 3, "z": 2},
        ],
        columns=SIGNAL_POINT_REQUIRED_COLUMNS,
    )
    pd.testing.assert_frame_equal(points.reset_index(drop=True), expected)


def test_build_signal_points_from_masks_writes_csv_and_space_json(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "out"
    mask_dir.mkdir()

    mask = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    tifffile.imwrite(mask_dir / "signal_1.tif", mask)

    build_signal_points_from_masks(
        mask_dir=mask_dir,
        out_dir=out_dir,
        subject_name="Test Subject",
        space_name="subject_space",
        orientation="las",
        resolution_um=[1.8, 1.8, 5.0],
        max_workers=1,
    )

    assert (out_dir / "Test_Subject_signal_points.csv").exists()
    assert (out_dir / "Test_Subject_signal_points_space.json").exists()
    points = pd.read_csv(out_dir / "Test_Subject_signal_points.csv")
    assert list(points["point_id"]) == [1, 2, 3]
    with (out_dir / "Test_Subject_signal_points_space.json").open(
        "r", encoding="utf-8"
    ) as handle:
        metadata = json.load(handle)
    assert metadata["representation"]["kind"] == "point_cloud"
    assert metadata["representation"]["representation_type"] == "signal_points"
