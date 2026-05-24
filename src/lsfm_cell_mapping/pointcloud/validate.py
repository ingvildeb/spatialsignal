"""Validation helpers for comparing Python and legacy MATLAB centroid tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lsfm_cell_mapping.pointcloud.centroids import POINTCLOUD_REQUIRED_COLUMNS


LEGACY_MATLAB_COLUMNS = ["seg_num", "x", "y", "z"]


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a Python vs MATLAB centroid-table comparison."""

    python_rows: int
    matlab_rows: int
    shared_rows: int
    exact_matches: int
    python_only_rows: int
    matlab_only_rows: int

    @property
    def all_rows_match(self) -> bool:
        """Return True if the tables match exactly after canonicalization."""

        return (
            self.python_rows == self.matlab_rows
            and self.exact_matches == self.shared_rows
            and self.python_only_rows == 0
            and self.matlab_only_rows == 0
        )


def load_canonical_pointcloud_csv(csv_path: Path) -> pd.DataFrame:
    """Load a canonical point-cloud CSV and validate its schema."""

    df = pd.read_csv(csv_path)
    expected = POINTCLOUD_REQUIRED_COLUMNS
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(
            f"Canonical CSV at {csv_path} is missing required columns {missing}"
        )

    return df.astype({column: int for column in expected})


def load_legacy_matlab_centroids_csv(csv_path: Path) -> pd.DataFrame:
    """Load the legacy MATLAB centroid CSV and convert it to canonical meaning.

    The MATLAB table uses headers ``seg_num,x,y,z``, but the stored values are:

    - seg_num
    - row
    - col
    - slice
    """

    df = pd.read_csv(csv_path)
    expected = LEGACY_MATLAB_COLUMNS
    if list(df.columns) != expected:
        raise ValueError(
            f"MATLAB CSV at {csv_path} has columns {list(df.columns)}, expected {expected}"
        )

    df = df.rename(columns={"x": "y", "y": "x", "z": "z"})
    return df[POINTCLOUD_REQUIRED_COLUMNS].astype({column: int for column in POINTCLOUD_REQUIRED_COLUMNS})


def relabel_slices_in_natural_order(df: pd.DataFrame) -> pd.DataFrame:
    """Relabel slice values to sequential order based on sorted unique values.

    This is useful when a legacy workflow stores filename-derived plane numbers
    rather than canonical sequential slice indices.
    """

    relabeled = df.copy()
    unique_slices = sorted(relabeled["z"].unique())
    slice_map = {original_slice: index for index, original_slice in enumerate(unique_slices, start=1)}
    relabeled["z"] = relabeled["z"].map(slice_map)
    return relabeled


def compare_pointcloud_tables(
    python_df: pd.DataFrame,
    matlab_df: pd.DataFrame,
) -> tuple[ValidationReport, pd.DataFrame, pd.DataFrame]:
    """Compare canonicalized Python and MATLAB centroid tables.

    Returns
    -------
    tuple
        ``(report, python_only, matlab_only)``, where the latter two tables
        contain rows that are not shared between the inputs.
    """

    python_norm = python_df.sort_values(POINTCLOUD_REQUIRED_COLUMNS).reset_index(drop=True)
    matlab_norm = matlab_df.sort_values(POINTCLOUD_REQUIRED_COLUMNS).reset_index(drop=True)

    python_counts = (
        python_norm.value_counts(subset=POINTCLOUD_REQUIRED_COLUMNS).rename("python_count").reset_index()
    )
    matlab_counts = (
        matlab_norm.value_counts(subset=POINTCLOUD_REQUIRED_COLUMNS).rename("matlab_count").reset_index()
    )

    merged = python_counts.merge(
        matlab_counts,
        on=POINTCLOUD_REQUIRED_COLUMNS,
        how="outer",
    ).fillna(0)

    merged["python_count"] = merged["python_count"].astype(int)
    merged["matlab_count"] = merged["matlab_count"].astype(int)

    exact_matches = int(
        merged.loc[merged["python_count"] == merged["matlab_count"], "python_count"].sum()
    )

    python_only = merged.loc[merged["python_count"] > merged["matlab_count"]].copy()
    python_only["difference"] = python_only["python_count"] - python_only["matlab_count"]

    matlab_only = merged.loc[merged["matlab_count"] > merged["python_count"]].copy()
    matlab_only["difference"] = matlab_only["matlab_count"] - matlab_only["python_count"]

    report = ValidationReport(
        python_rows=len(python_norm),
        matlab_rows=len(matlab_norm),
        shared_rows=min(len(python_norm), len(matlab_norm)),
        exact_matches=exact_matches,
        python_only_rows=int(python_only["difference"].sum()) if not python_only.empty else 0,
        matlab_only_rows=int(matlab_only["difference"].sum()) if not matlab_only.empty else 0,
    )

    return report, python_only, matlab_only
