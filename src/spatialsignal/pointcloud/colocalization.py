"""Within-plane colocalization matching for centroid point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from spatialsignal.models import PointCloudDataset, SpaceDefinition


MATCH_COLUMNS = [
    "a_detection_id",
    "b_detection_id",
    "z",
    "dx_um",
    "dy_um",
    "xy_distance_um",
    "a_candidate_count",
    "b_candidate_count",
    "is_ambiguous",
]

PLANE_SUMMARY_COLUMNS = [
    "z",
    "n_a_detections",
    "n_b_detections",
    "n_candidate_pairs",
    "n_accepted_matches",
    "n_unmatched_a",
    "n_unmatched_b",
    "raw_match_rate_a",
    "raw_match_rate_b",
    "median_match_distance_um",
    "maximum_match_distance_um",
    "n_ambiguous_a",
    "n_ambiguous_b",
]

OBJECT_RELATIONSHIP_COLUMNS = [
    "a_object_id",
    "b_object_id",
    "n_detection_matches",
    "n_matched_planes",
    "min_distance_um",
    "median_distance_um",
    "max_distance_um",
    "a_partner_count",
    "b_partner_count",
    "relationship_status",
]


@dataclass(frozen=True)
class ColocalizationResult:
    """Accepted detection matches and per-plane matching QC."""

    matches: pd.DataFrame
    plane_summary: pd.DataFrame


@dataclass(frozen=True)
class ObjectRelationshipResult:
    """Detection-level evidence lifted to deduplicated object relationships."""

    relationships: pd.DataFrame


def match_colocalized_detections(
    dataset_a: PointCloudDataset,
    dataset_b: PointCloudDataset,
    *,
    max_xy_distance_um: float,
) -> ColocalizationResult:
    """Match A and B centroid detections one-to-one within each exact z plane.

    Matching is lexicographic: maximize the number of accepted pairs first,
    then minimize their total lateral distance. Z resolution never enters the
    distance calculation because detections are only compared at equal z.
    """

    if max_xy_distance_um < 0:
        raise ValueError(
            "max_xy_distance_um must be >= 0, "
            f"got {max_xy_distance_um}"
        )

    _validate_colocalization_inputs(dataset_a, dataset_b)
    x_res_um, y_res_um = _xy_resolution(dataset_a.space)
    a_x, a_y = _preferred_xy_columns(dataset_a.points)
    b_x, b_y = _preferred_xy_columns(dataset_b.points)

    a_by_z = {
        int(z): table.sort_values("detection_id").reset_index(drop=True)
        for z, table in dataset_a.points.groupby("z", sort=True)
    }
    b_by_z = {
        int(z): table.sort_values("detection_id").reset_index(drop=True)
        for z, table in dataset_b.points.groupby("z", sort=True)
    }

    match_tables: list[pd.DataFrame] = []
    plane_rows: list[dict[str, int | float]] = []
    for z in _declared_z_indices(dataset_a.space):
        a_points = a_by_z.get(z, dataset_a.points.iloc[0:0])
        b_points = b_by_z.get(z, dataset_b.points.iloc[0:0])
        candidates = _build_plane_candidates(
            a_points,
            b_points,
            a_x=a_x,
            a_y=a_y,
            b_x=b_x,
            b_y=b_y,
            x_res_um=x_res_um,
            y_res_um=y_res_um,
            max_xy_distance_um=max_xy_distance_um,
        )
        matches = _select_plane_matches(
            candidates,
            max_xy_distance_um=max_xy_distance_um,
        )
        if not matches.empty:
            matches.insert(2, "z", int(z))
            match_tables.append(matches[MATCH_COLUMNS])

        n_a = int(len(a_points))
        n_b = int(len(b_points))
        n_matches = int(len(matches))
        distances = matches["xy_distance_um"] if n_matches else pd.Series(dtype=float)
        plane_rows.append(
            {
                "z": int(z),
                "n_a_detections": n_a,
                "n_b_detections": n_b,
                "n_candidate_pairs": int(len(candidates)),
                "n_accepted_matches": n_matches,
                "n_unmatched_a": n_a - n_matches,
                "n_unmatched_b": n_b - n_matches,
                "raw_match_rate_a": n_matches / n_a if n_a else np.nan,
                "raw_match_rate_b": n_matches / n_b if n_b else np.nan,
                "median_match_distance_um": (
                    float(distances.median()) if n_matches else np.nan
                ),
                "maximum_match_distance_um": (
                    float(distances.max()) if n_matches else np.nan
                ),
                "n_ambiguous_a": int(
                    candidates.loc[
                        candidates["a_candidate_count"] > 1,
                        "a_detection_id",
                    ].nunique()
                )
                if not candidates.empty
                else 0,
                "n_ambiguous_b": int(
                    candidates.loc[
                        candidates["b_candidate_count"] > 1,
                        "b_detection_id",
                    ].nunique()
                )
                if not candidates.empty
                else 0,
            }
        )

    all_matches = (
        pd.concat(match_tables, ignore_index=True)
        if match_tables
        else pd.DataFrame(columns=MATCH_COLUMNS)
    )
    plane_summary = pd.DataFrame(plane_rows, columns=PLANE_SUMMARY_COLUMNS)
    return ColocalizationResult(matches=all_matches, plane_summary=plane_summary)


def map_detection_matches_to_objects(
    detection_matches: pd.DataFrame,
    a_membership: pd.DataFrame,
    b_membership: pd.DataFrame,
) -> ObjectRelationshipResult:
    """Map accepted raw matches through channel-specific object memberships."""

    _require_columns(detection_matches, MATCH_COLUMNS, "detection_matches")
    _validate_membership(a_membership, "a_membership")
    _validate_membership(b_membership, "b_membership")

    if detection_matches.empty:
        return ObjectRelationshipResult(
            relationships=pd.DataFrame(columns=OBJECT_RELATIONSHIP_COLUMNS)
        )

    a_lookup = a_membership.rename(
        columns={"detection_id": "a_detection_id", "object_id": "a_object_id"}
    )
    b_lookup = b_membership.rename(
        columns={"detection_id": "b_detection_id", "object_id": "b_object_id"}
    )
    lifted = detection_matches.merge(
        a_lookup,
        on="a_detection_id",
        how="left",
        validate="many_to_one",
    ).merge(
        b_lookup,
        on="b_detection_id",
        how="left",
        validate="many_to_one",
    )
    if lifted[["a_object_id", "b_object_id"]].isna().any().any():
        raise ValueError(
            "Every matched detection must occur in its channel's membership table"
        )

    relationships = (
        lifted.groupby(["a_object_id", "b_object_id"], sort=True, observed=True)
        .agg(
            n_detection_matches=("z", "size"),
            n_matched_planes=("z", "nunique"),
            min_distance_um=("xy_distance_um", "min"),
            median_distance_um=("xy_distance_um", "median"),
            max_distance_um=("xy_distance_um", "max"),
        )
        .reset_index()
    )
    relationships[["a_object_id", "b_object_id"]] = relationships[
        ["a_object_id", "b_object_id"]
    ].astype(np.int64)
    a_partner_counts = relationships.groupby("a_object_id")["b_object_id"].transform(
        "nunique"
    )
    b_partner_counts = relationships.groupby("b_object_id")["a_object_id"].transform(
        "nunique"
    )
    relationships["a_partner_count"] = a_partner_counts.astype(np.int64)
    relationships["b_partner_count"] = b_partner_counts.astype(np.int64)
    relationships["relationship_status"] = [
        _relationship_status(int(a_count), int(b_count))
        for a_count, b_count in zip(a_partner_counts, b_partner_counts, strict=True)
    ]
    return ObjectRelationshipResult(
        relationships=relationships[OBJECT_RELATIONSHIP_COLUMNS]
    )


def _build_plane_candidates(
    a_points: pd.DataFrame,
    b_points: pd.DataFrame,
    *,
    a_x: str,
    a_y: str,
    b_x: str,
    b_y: str,
    x_res_um: float,
    y_res_um: float,
    max_xy_distance_um: float,
) -> pd.DataFrame:
    columns = [
        "a_detection_id",
        "b_detection_id",
        "dx_um",
        "dy_um",
        "xy_distance_um",
        "a_candidate_count",
        "b_candidate_count",
    ]
    if a_points.empty or b_points.empty:
        return pd.DataFrame(columns=columns)

    a_xy_um = np.column_stack(
        (
            a_points[a_x].to_numpy(dtype=float) * x_res_um,
            a_points[a_y].to_numpy(dtype=float) * y_res_um,
        )
    )
    b_xy_um = np.column_stack(
        (
            b_points[b_x].to_numpy(dtype=float) * x_res_um,
            b_points[b_y].to_numpy(dtype=float) * y_res_um,
        )
    )
    neighbor_lists = cKDTree(b_xy_um).query_ball_point(
        a_xy_um,
        r=max_xy_distance_um,
    )
    a_ids = a_points["detection_id"].to_numpy(dtype=np.int64)
    b_ids = b_points["detection_id"].to_numpy(dtype=np.int64)
    rows: list[dict[str, int | float]] = []
    for a_index, b_indices in enumerate(neighbor_lists):
        for b_index in b_indices:
            dx_um = float(b_xy_um[b_index, 0] - a_xy_um[a_index, 0])
            dy_um = float(b_xy_um[b_index, 1] - a_xy_um[a_index, 1])
            rows.append(
                {
                    "a_detection_id": int(a_ids[a_index]),
                    "b_detection_id": int(b_ids[b_index]),
                    "dx_um": dx_um,
                    "dy_um": dy_um,
                    "xy_distance_um": float(np.hypot(dx_um, dy_um)),
                }
            )
    candidates = pd.DataFrame(rows)
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates["a_candidate_count"] = candidates.groupby("a_detection_id")[
        "b_detection_id"
    ].transform("size")
    candidates["b_candidate_count"] = candidates.groupby("b_detection_id")[
        "a_detection_id"
    ].transform("size")
    return candidates[columns].sort_values(
        ["a_detection_id", "b_detection_id"]
    ).reset_index(drop=True)


def _select_plane_matches(
    candidates: pd.DataFrame,
    *,
    max_xy_distance_um: float,
) -> pd.DataFrame:
    output_columns = [column for column in MATCH_COLUMNS if column != "z"]
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)

    selected_pairs: set[tuple[int, int]] = set()
    for a_ids, b_ids in _candidate_components(candidates):
        component = candidates.loc[
            candidates["a_detection_id"].isin(a_ids)
            & candidates["b_detection_id"].isin(b_ids)
        ]
        a_ids = sorted(a_ids)
        b_ids = sorted(b_ids)
        a_positions = {detection_id: index for index, detection_id in enumerate(a_ids)}
        b_positions = {detection_id: index for index, detection_id in enumerate(b_ids)}
        n_a = len(a_ids)
        n_b = len(b_ids)
        size = n_a + n_b
        unmatched_cost = (min(n_a, n_b) + 1) * (max_xy_distance_um + 1.0)
        forbidden_cost = unmatched_cost * (size + 2)
        costs = np.full((size, size), forbidden_cost, dtype=float)
        costs[:n_a, n_b:] = unmatched_cost
        costs[n_a:, :n_b] = unmatched_cost
        costs[n_a:, n_b:] = 0.0
        for row in component.itertuples(index=False):
            costs[
                a_positions[int(row.a_detection_id)],
                b_positions[int(row.b_detection_id)],
            ] = float(row.xy_distance_um)

        row_indices, column_indices = linear_sum_assignment(costs)
        for row_index, column_index in zip(row_indices, column_indices, strict=True):
            if row_index < n_a and column_index < n_b:
                if costs[row_index, column_index] < forbidden_cost:
                    selected_pairs.add((a_ids[row_index], b_ids[column_index]))

    pair_index = pd.MultiIndex.from_tuples(
        sorted(selected_pairs),
        names=["a_detection_id", "b_detection_id"],
    )
    indexed = candidates.set_index(["a_detection_id", "b_detection_id"])
    matches = indexed.loc[pair_index].reset_index()
    matches["is_ambiguous"] = (
        (matches["a_candidate_count"] > 1)
        | (matches["b_candidate_count"] > 1)
    )
    return matches[output_columns]


def _candidate_components(
    candidates: pd.DataFrame,
) -> list[tuple[set[int], set[int]]]:
    a_to_b: dict[int, set[int]] = {}
    b_to_a: dict[int, set[int]] = {}
    for row in candidates.itertuples(index=False):
        a_id = int(row.a_detection_id)
        b_id = int(row.b_detection_id)
        a_to_b.setdefault(a_id, set()).add(b_id)
        b_to_a.setdefault(b_id, set()).add(a_id)

    components: list[tuple[set[int], set[int]]] = []
    unseen_a = set(a_to_b)
    while unseen_a:
        start = min(unseen_a)
        component_a: set[int] = set()
        component_b: set[int] = set()
        pending: list[tuple[str, int]] = [("a", start)]
        while pending:
            side, node = pending.pop()
            if side == "a":
                if node in component_a:
                    continue
                component_a.add(node)
                pending.extend(("b", neighbor) for neighbor in a_to_b[node])
            else:
                if node in component_b:
                    continue
                component_b.add(node)
                pending.extend(("a", neighbor) for neighbor in b_to_a[node])
        unseen_a.difference_update(component_a)
        components.append((component_a, component_b))
    return components


def _validate_colocalization_inputs(
    dataset_a: PointCloudDataset,
    dataset_b: PointCloudDataset,
) -> None:
    dataset_a.validate()
    dataset_b.validate()
    if dataset_a.subject_name != dataset_b.subject_name:
        raise ValueError(
            "Point clouds must have the same subject_name: "
            f"{dataset_a.subject_name!r} vs {dataset_b.subject_name!r}"
        )
    for label, dataset in (("A", dataset_a), ("B", dataset_b)):
        representation = dataset.metadata.representation
        if representation.kind != "point_cloud" or representation.representation_type != "point_centroids":
            raise ValueError(
                f"Dataset {label} must have representation_type='point_centroids'"
            )
        if dataset.points["detection_id"].duplicated().any():
            raise ValueError(f"Dataset {label} detection_id values must be unique")

    a_space = dataset_a.space
    b_space = dataset_b.space
    differing = []
    for field in ("orientation", "axis_labels", "indexing", "units", "shape"):
        if getattr(a_space, field) != getattr(b_space, field):
            differing.append(field)
    if not np.allclose(a_space.resolution_um, b_space.resolution_um):
        differing.append("resolution_um")
    if differing:
        raise ValueError(
            "Point clouds occupy incompatible spaces; differing fields: "
            + ", ".join(differing)
        )


def _declared_z_indices(space: SpaceDefinition) -> range:
    z_axis = space.axis_labels.index("z")
    z_size = int(space.shape[z_axis])
    start = 0 if space.indexing == "zero_based" else 1
    return range(start, start + z_size)


def _xy_resolution(space: SpaceDefinition) -> tuple[float, float]:
    resolution = dict(zip(space.axis_labels, space.resolution_um, strict=True))
    return float(resolution["x"]), float(resolution["y"])


def _preferred_xy_columns(points: pd.DataFrame) -> tuple[str, str]:
    return (
        "x_float" if "x_float" in points.columns else "x",
        "y_float" if "y_float" in points.columns else "y",
    )


def _validate_membership(membership: pd.DataFrame, name: str) -> None:
    _require_columns(membership, ["object_id", "detection_id"], name)
    if membership["detection_id"].duplicated().any():
        raise ValueError(f"{name} must assign each detection_id exactly once")


def _require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _relationship_status(a_partner_count: int, b_partner_count: int) -> str:
    if a_partner_count == 1 and b_partner_count == 1:
        return "one_to_one"
    if a_partner_count > 1 and b_partner_count == 1:
        return "one_a_to_multiple_b"
    if a_partner_count == 1 and b_partner_count > 1:
        return "multiple_a_to_one_b"
    return "many_to_many"
