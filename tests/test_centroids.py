from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from lsfm_cell_mapping.pointcloud import (
    POINTCLOUD_COLUMNS,
    extract_centroids_from_mask,
    extract_centroids_from_mask_stack,
    matlab_round,
)


def test_matlab_round_matches_expected_half_up_behavior() -> None:
    assert matlab_round(10.5) == 11
    assert matlab_round(11.5) == 12
    assert matlab_round(10.49) == 10


def test_extract_centroids_from_mask_returns_expected_rows(tmp_path: Path) -> None:
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
            {"seg_num": 1, "row": 1, "col": 3, "slice": 1},
            {"seg_num": 2, "row": 3, "col": 2, "slice": 1},
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

    expected = pd.DataFrame(
        [
            {"seg_num": 1, "row": 1, "col": 3, "slice": 1},
            {"seg_num": 2, "row": 3, "col": 2, "slice": 1},
            {"seg_num": 3, "row": 2, "col": 3, "slice": 2},
        ],
        columns=POINTCLOUD_COLUMNS,
    )

    pd.testing.assert_frame_equal(pointcloud.reset_index(drop=True), expected)
