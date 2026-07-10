from pathlib import Path

import pandas as pd

from spatialsignal.pointcloud import (
    POINTCLOUD_REQUIRED_COLUMNS,
    compare_pointcloud_tables,
    load_canonical_pointcloud_csv,
    load_legacy_matlab_centroids_csv,
    relabel_slices_in_natural_order,
)


def test_load_legacy_matlab_centroids_csv_maps_columns_to_canonical(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "centroids.csv"
    pd.DataFrame(
        [
            {"seg_num": 1, "x": 10, "y": 20, "z": 4},
            {"seg_num": 2, "x": 30, "y": 40, "z": 8},
        ]
    ).to_csv(csv_path, index=False)

    loaded = load_legacy_matlab_centroids_csv(csv_path)

    expected = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 20, "y": 10, "z": 4},
            {"detection_id": 2, "seg_num": 2, "x": 40, "y": 30, "z": 8},
        ],
        columns=POINTCLOUD_REQUIRED_COLUMNS,
    )

    pd.testing.assert_frame_equal(loaded, expected)


def test_relabel_slices_in_natural_order_maps_unique_sorted_values() -> None:
    df = pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 20, "y": 10, "z": 4},
            {"detection_id": 2, "seg_num": 2, "x": 40, "y": 30, "z": 12},
            {"detection_id": 3, "seg_num": 3, "x": 60, "y": 50, "z": 8},
        ]
    )

    relabeled = relabel_slices_in_natural_order(df)

    assert list(relabeled["z"]) == [1, 3, 2]


def test_compare_pointcloud_tables_reports_full_match(tmp_path: Path) -> None:
    python_csv = tmp_path / "pointcloud.csv"
    matlab_csv = tmp_path / "centroids.csv"

    pd.DataFrame(
        [
            {"detection_id": 1, "seg_num": 1, "x": 20, "y": 10, "z": 1},
            {"detection_id": 2, "seg_num": 2, "x": 40, "y": 30, "z": 2},
        ]
    , columns=POINTCLOUD_REQUIRED_COLUMNS).to_csv(python_csv, index=False)

    pd.DataFrame(
        [
            {"seg_num": 1, "x": 10, "y": 20, "z": 4},
            {"seg_num": 2, "x": 30, "y": 40, "z": 8},
        ]
    ).to_csv(matlab_csv, index=False)

    python_df = load_canonical_pointcloud_csv(python_csv)
    matlab_df = relabel_slices_in_natural_order(load_legacy_matlab_centroids_csv(matlab_csv))

    report, python_only, matlab_only = compare_pointcloud_tables(python_df, matlab_df)

    assert report.all_rows_match
    assert python_only.empty
    assert matlab_only.empty
