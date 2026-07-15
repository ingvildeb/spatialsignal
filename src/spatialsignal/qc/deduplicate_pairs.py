"""Quality-control helpers for cross-plane deduplication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spatialsignal.io.masks import find_mask_files
from spatialsignal.utils.images import read_2d_mask


def select_qc_plane_pairs(
    points: pd.DataFrame,
    *,
    n_pairs: int = 3,
    edge_exclusion_fraction: float = 0.1,
) -> list[tuple[int, int]]:
    """Select informative adjacent plane pairs for deduplication QC."""

    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    if not 0 <= edge_exclusion_fraction < 0.5:
        raise ValueError(
            "edge_exclusion_fraction must be in [0, 0.5)"
        )

    counts = points.groupby("z").size().sort_index()
    z_values = list(counts.index.astype(int))
    if len(z_values) < 2:
        return []

    n_exclude = int(len(z_values) * edge_exclusion_fraction)
    eligible = z_values[n_exclude : len(z_values) - n_exclude] if n_exclude else z_values[:]
    if len(eligible) < 2:
        eligible = z_values[:]

    eligible_pairs = []
    eligible_set = set(eligible)
    for z in eligible:
        if z + 1 in eligible_set:
            eligible_pairs.append((z, z + 1))
    if not eligible_pairs:
        eligible_pairs = [(z_values[i], z_values[i + 1]) for i in range(len(z_values) - 1)]

    target_positions = np.linspace(0.25, 0.75, n_pairs) if n_pairs > 1 else np.array([0.5])
    z_min = eligible_pairs[0][0]
    z_max = eligible_pairs[-1][1]
    selected: list[tuple[int, int]] = []
    used = set()

    for target_fraction in target_positions:
        target_center = z_min + target_fraction * (z_max - z_min)
        best_pair = None
        best_key = None
        for pair in eligible_pairs:
            if pair in used:
                continue
            pair_center = (pair[0] + pair[1]) / 2
            combined_count = int(counts.loc[pair[0]] + counts.loc[pair[1]])
            key = (abs(pair_center - target_center), -combined_count, pair[0])
            if best_key is None or key < best_key:
                best_key = key
                best_pair = pair
        if best_pair is not None:
            selected.append(best_pair)
            used.add(best_pair)

    return selected


def write_pair_duplicate_qc(
    *,
    points: pd.DataFrame,
    membership: pd.DataFrame,
    mask_dir: Path,
    plane_pairs: list[tuple[int, int]],
    output_dir: Path,
    extensions: tuple[str, ...] = (".tif", ".tiff"),
    prefix: str | None = None,
    suffix: str | None = None,
) -> Path:
    """Write pairwise duplicate-vs-nonduplicate mask QC images."""

    mask_files = find_mask_files(
        mask_dir,
        extensions=extensions,
        prefix=prefix,
        suffix=suffix,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    z_values = sorted(points["z"].astype(int).unique())
    z_to_mask = {
        z: mask_files[index]
        for index, z in enumerate(z_values)
        if index < len(mask_files)
    }

    summary: dict[str, object] = {
        "plane_pairs": [list(pair) for pair in plane_pairs],
        "mask_dir": str(mask_dir),
        "duplicate_color_rgb": [0, 255, 0],
        "nonduplicate_color_rgb": [255, 0, 0],
        "pair_summaries": [],
    }

    for plane_a, plane_b in plane_pairs:
        if plane_a not in z_to_mask or plane_b not in z_to_mask:
            raise ValueError(
                f"Plane pair ({plane_a}, {plane_b}) cannot be mapped to mask files from {mask_dir}"
            )

        pair_summary = _write_single_pair_qc(
            points=points,
            membership=membership,
            plane_a=plane_a,
            plane_b=plane_b,
            mask_a_path=z_to_mask[plane_a],
            mask_b_path=z_to_mask[plane_b],
            output_dir=output_dir,
        )
        summary["pair_summaries"].append(pair_summary)

    summary_path = output_dir / "pair_qc_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return summary_path


def _write_single_pair_qc(
    *,
    points: pd.DataFrame,
    membership: pd.DataFrame,
    plane_a: int,
    plane_b: int,
    mask_a_path: Path,
    mask_b_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    selected_planes = {plane_a, plane_b}
    plane_points = points.loc[points["z"].isin(selected_planes), ["detection_id", "seg_num", "z"]].copy()
    selected_detection_ids = set(plane_points["detection_id"].astype(int).tolist())
    plane_membership = membership.loc[membership["detection_id"].isin(selected_detection_ids)].copy()
    plane_subset = plane_membership.merge(plane_points, on="detection_id", how="left", validate="many_to_one")
    object_plane_sets = plane_subset.groupby("object_id")["z"].agg(lambda s: set(map(int, s.tolist())))
    pair_object_ids = set(object_plane_sets.loc[object_plane_sets == selected_planes].index.tolist())

    mask_a = read_2d_mask(mask_a_path)
    mask_b = read_2d_mask(mask_b_path)
    _validate_plane_mask_alignment(mask_a, plane_points, plane_a)
    _validate_plane_mask_alignment(mask_b, plane_points, plane_b)

    out_a = output_dir / f"{mask_a_path.stem}_pair_duplicate_qc.tif"
    out_b = output_dir / f"{mask_b_path.stem}_pair_duplicate_qc.tif"
    plane_a_summary = _color_single_plane_mask(mask_a, plane_subset, pair_object_ids, plane_a, mask_a_path, out_a)
    plane_b_summary = _color_single_plane_mask(mask_b, plane_subset, pair_object_ids, plane_b, mask_b_path, out_b)

    return {
        "planes": [plane_a, plane_b],
        "pair_object_count": int(len(pair_object_ids)),
        "plane_summaries": {
            str(plane_a): plane_a_summary,
            str(plane_b): plane_b_summary,
        },
    }


def _validate_plane_mask_alignment(mask: np.ndarray, plane_points: pd.DataFrame, plane: int) -> None:
    mask_label_count = int(len(np.unique(mask)) - (1 if 0 in mask else 0))
    point_label_count = int(plane_points.loc[plane_points["z"] == plane, "seg_num"].nunique())
    if mask_label_count != point_label_count:
        raise ValueError(
            f"Mask/point-cloud mismatch for z={plane}: mask has {mask_label_count} labels but point cloud has {point_label_count}"
        )


def _color_single_plane_mask(
    mask: np.ndarray,
    plane_subset: pd.DataFrame,
    pair_object_ids: set[int],
    plane: int,
    input_mask_path: Path,
    output_path: Path,
) -> dict[str, object]:
    plane_rows = plane_subset.loc[plane_subset["z"] == plane, ["seg_num", "object_id"]].drop_duplicates()
    duplicate_seg_nums = plane_rows.loc[
        plane_rows["object_id"].isin(pair_object_ids), "seg_num"
    ].astype(mask.dtype).to_numpy()
    all_seg_nums = plane_rows["seg_num"].astype(mask.dtype).to_numpy()
    nonduplicate_seg_nums = np.setdiff1d(all_seg_nums, duplicate_seg_nums, assume_unique=False)

    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[np.isin(mask, nonduplicate_seg_nums)] = (255, 0, 0)
    rgb[np.isin(mask, duplicate_seg_nums)] = (0, 255, 0)
    tifffile.imwrite(output_path, rgb)

    return {
        "input_mask": str(input_mask_path),
        "output_qc": str(output_path),
        "duplicate_seg_num_count": int(len(np.unique(duplicate_seg_nums))),
        "nonduplicate_seg_num_count": int(len(np.unique(nonduplicate_seg_nums))),
    }
