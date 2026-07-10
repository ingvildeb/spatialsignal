"""Example runner for comparing canonical Python and legacy MATLAB centroid CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from spatialsignal.pointcloud import (
    compare_pointcloud_tables,
    load_canonical_pointcloud_csv,
    load_legacy_matlab_centroids_csv,
    relabel_slices_in_natural_order,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for CSV validation."""

    parser = argparse.ArgumentParser(
        description="Compare canonical Python point-cloud CSV output against legacy MATLAB centroids.csv."
    )
    parser.add_argument(
        "--python-csv",
        required=True,
        help="Path to the canonical Python CSV with required columns detection_id,seg_num,x,y,z.",
    )
    parser.add_argument(
        "--matlab-csv",
        required=True,
        help="Path to the legacy MATLAB centroids.csv with columns seg_num,x,y,z.",
    )
    parser.add_argument(
        "--show-differences",
        type=int,
        default=10,
        help="Maximum number of differing rows to print from each side.",
    )
    parser.add_argument(
        "--relabel-matlab-slices",
        action="store_true",
        help=(
            "Relabel MATLAB slice values to natural sequential order before comparison. "
            "Useful when the MATLAB CSV stores filename-derived plane numbers."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Compare Python and MATLAB centroid CSV outputs and print a summary."""

    args = parse_args()
    python_df = load_canonical_pointcloud_csv(Path(args.python_csv))
    matlab_df = load_legacy_matlab_centroids_csv(Path(args.matlab_csv))
    if args.relabel_matlab_slices:
        matlab_df = relabel_slices_in_natural_order(matlab_df)

    report, python_only, matlab_only = compare_pointcloud_tables(python_df, matlab_df)

    print("Validation summary")
    print(f"  Python rows:        {report.python_rows}")
    print(f"  MATLAB rows:        {report.matlab_rows}")
    print(f"  Exact matched rows: {report.exact_matches}")
    print(f"  Python-only rows:   {report.python_only_rows}")
    print(f"  MATLAB-only rows:   {report.matlab_only_rows}")
    print(f"  All rows match:     {report.all_rows_match}")

    limit = max(args.show_differences, 0)
    if limit and not python_only.empty:
        print("\nPython-only differences")
        print(python_only.head(limit).to_string(index=False))

    if limit and not matlab_only.empty:
        print("\nMATLAB-only differences")
        print(matlab_only.head(limit).to_string(index=False))

    return 0 if report.all_rows_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
