"""Cross-plane deduplication for raw point-cloud detections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spatialsignal.models.datasets import _validate_required_columns
from spatialsignal.models.metadata import SpaceDefinition


EDGE_COLUMNS = [
    "source_detection_id",
    "target_detection_id",
    "source_z",
    "target_z",
    "plane_offset",
    "dx_um",
    "dy_um",
    "xy_distance_um",
    "dz_um",
]

CLEANED_OBJECT_REQUIRED_COLUMNS = [
    "object_id",
    "x",
    "y",
    "z",
    "x_float",
    "y_float",
    "z_float",
    "n_detections",
    "n_planes",
    "z_min",
    "z_max",
]

_OPTIONAL_OBJECT_AGGREGATIONS = {
    "mean_area_px": ("area_px", "mean"),
    "max_area_px": ("area_px", "max"),
    "mean_major_axis_length_px": ("major_axis_length_px", "mean"),
    "mean_minor_axis_length_px": ("minor_axis_length_px", "mean"),
    "mean_eccentricity": ("eccentricity", "mean"),
}


@dataclass(frozen=True)
class DeduplicationResult:
    """Container for cross-plane deduplication outputs."""

    edges: pd.DataFrame
    membership: pd.DataFrame
    objects: pd.DataFrame


def summarize_deduplication_result(
    raw_points: pd.DataFrame,
    result: DeduplicationResult,
) -> dict[str, int]:
    """Return a compact summary of deduplication outputs."""

    return {
        "raw_detections": int(len(raw_points)),
        "cleaned_objects": int(len(result.objects)),
        "accepted_edges": int(len(result.edges)),
    }


def deduplicate_across_planes(
    points: pd.DataFrame,
    space: SpaceDefinition,
    *,
    max_plane_offset: int,
    max_xy_distance_um: float,
    max_n_planes: int | None = None,
) -> DeduplicationResult:
    """Merge raw detections that likely represent the same cell across planes.

    Every cleaned object is constrained to contain at most one detection from
    any given plane.
    """

    if max_plane_offset < 1:
        raise ValueError(f"max_plane_offset must be >= 1, got {max_plane_offset}")
    if max_xy_distance_um < 0:
        raise ValueError(f"max_xy_distance_um must be >= 0, got {max_xy_distance_um}")
    if max_n_planes is not None and max_n_planes < 1:
        raise ValueError(f"max_n_planes must be >= 1 when provided, got {max_n_planes}")

    _validate_required_columns(points)

    candidate_edges = build_cross_plane_edge_table(
        points,
        space,
        max_plane_offset=max_plane_offset,
        max_xy_distance_um=max_xy_distance_um,
    )
    membership, accepted_edges = build_object_membership_table(
        points,
        candidate_edges,
        max_n_planes=max_n_planes,
    )
    objects = aggregate_cleaned_objects(points, membership)
    invalid = objects["n_detections"] != objects["n_planes"]
    if invalid.any():
        invalid_ids = objects.loc[invalid, "object_id"].astype(int).tolist()
        raise RuntimeError(
            "Deduplication produced objects with multiple detections from the "
            f"same plane: object IDs {invalid_ids[:10]}"
        )
    return DeduplicationResult(edges=accepted_edges, membership=membership, objects=objects)


def build_cross_plane_edge_table(
    points: pd.DataFrame,
    space: SpaceDefinition,
    *,
    max_plane_offset: int,
    max_xy_distance_um: float,
) -> pd.DataFrame:
    """Build an edge table linking plausible same-cell detections across planes."""

    _validate_required_columns(points)
    x_res_um, y_res_um, z_res_um = _resolution_by_axis(space)
    x_coord, y_coord = _xy_coordinate_columns(points)

    edges: list[dict[str, int | float]] = []
    grouped = {int(z): df.copy() for z, df in points.groupby("z", sort=True)}
    z_values = sorted(grouped)

    for source_z in z_values:
        source_points = grouped[source_z]
        for plane_offset in range(1, max_plane_offset + 1):
            target_z = source_z + plane_offset
            if target_z not in grouped:
                continue

            target_points = grouped[target_z]
            source_xy_um = np.column_stack(
                (
                    source_points[x_coord].to_numpy(dtype=float) * x_res_um,
                    source_points[y_coord].to_numpy(dtype=float) * y_res_um,
                )
            )
            target_xy_um = np.column_stack(
                (
                    target_points[x_coord].to_numpy(dtype=float) * x_res_um,
                    target_points[y_coord].to_numpy(dtype=float) * y_res_um,
                )
            )
            target_tree = cKDTree(target_xy_um)
            neighbor_lists = target_tree.query_ball_point(source_xy_um, r=max_xy_distance_um)

            source_detection_ids = source_points["detection_id"].to_numpy(dtype=int)
            target_detection_ids = target_points["detection_id"].to_numpy(dtype=int)

            for source_index, target_indices in enumerate(neighbor_lists):
                if not target_indices:
                    continue

                source_x_um, source_y_um = source_xy_um[source_index]
                source_detection_id = int(source_detection_ids[source_index])
                for target_index in target_indices:
                    target_x_um, target_y_um = target_xy_um[target_index]
                    dx_um = float(target_x_um - source_x_um)
                    dy_um = float(target_y_um - source_y_um)
                    xy_distance_um = float(np.hypot(dx_um, dy_um))

                    edges.append(
                        {
                            "source_detection_id": source_detection_id,
                            "target_detection_id": int(target_detection_ids[target_index]),
                            "source_z": source_z,
                            "target_z": target_z,
                            "plane_offset": plane_offset,
                            "dx_um": dx_um,
                            "dy_um": dy_um,
                            "xy_distance_um": xy_distance_um,
                            "dz_um": plane_offset * z_res_um,
                        }
                    )

    return pd.DataFrame(edges, columns=EDGE_COLUMNS)


def build_object_membership_table(
    points: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    max_n_planes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert an edge table into object memberships via connected components."""

    detection_ids = [int(detection_id) for detection_id in points["detection_id"]]
    detection_to_z = {
        int(detection_id): int(z)
        for detection_id, z in zip(points["detection_id"], points["z"], strict=True)
    }
    components, accepted_edges = _connected_components(
        detection_ids,
        detection_to_z,
        edges,
        max_n_planes=max_n_planes,
    )

    rows: list[dict[str, int]] = []
    for object_id, component in enumerate(components, start=1):
        for detection_id in component:
            rows.append(
                {
                    "object_id": object_id,
                    "detection_id": detection_id,
                }
            )

    membership = pd.DataFrame(rows, columns=["object_id", "detection_id"])
    membership = membership.sort_values(["object_id", "detection_id"]).reset_index(drop=True)
    return membership, accepted_edges.reset_index(drop=True)


def aggregate_cleaned_objects(
    points: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate raw detections into cleaned objects with grouped reductions."""

    merged = membership.merge(
        points,
        on="detection_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if "seg_num" in merged.columns and merged["seg_num"].isna().any():
        raise ValueError("Membership contains detection IDs absent from the point cloud")

    x_coord, y_coord = _xy_coordinate_columns(points)
    aggregations: dict[str, tuple[str, str]] = {
        "x_float": (x_coord, "mean"),
        "y_float": (y_coord, "mean"),
        "z_float": ("z", "mean"),
        "n_detections": ("detection_id", "size"),
        "n_planes": ("z", "nunique"),
        "z_min": ("z", "min"),
        "z_max": ("z", "max"),
    }
    for output_column, (source_column, reducer) in _OPTIONAL_OBJECT_AGGREGATIONS.items():
        if source_column in points.columns:
            aggregations[output_column] = (source_column, reducer)

    membership_is_sorted = membership["object_id"].is_monotonic_increasing
    objects = (
        merged.groupby(
            "object_id",
            sort=not membership_is_sorted,
            observed=True,
        )
        .agg(**aggregations)
        .reset_index()
    )
    objects["x"] = np.floor(objects["x_float"] + 0.5).astype(np.int64)
    objects["y"] = np.floor(objects["y_float"] + 0.5).astype(np.int64)
    objects["z"] = np.floor(objects["z_float"] + 0.5).astype(np.int64)

    integer_columns = [
        "object_id",
        "x",
        "y",
        "z",
        "n_detections",
        "n_planes",
        "z_min",
        "z_max",
    ]
    objects[integer_columns] = objects[integer_columns].astype(np.int64)
    float_columns = [
        column
        for column in (
            "x_float",
            "y_float",
            "z_float",
            *_OPTIONAL_OBJECT_AGGREGATIONS,
        )
        if column in objects.columns
    ]
    objects[float_columns] = objects[float_columns].astype(np.float64)

    optional = [
        column for column in _OPTIONAL_OBJECT_AGGREGATIONS if column in objects.columns
    ]
    return objects[CLEANED_OBJECT_REQUIRED_COLUMNS + optional]


def _xy_coordinate_columns(points: pd.DataFrame) -> tuple[str, str]:
    """Return the preferred x/y coordinate columns for matching and aggregation."""

    x_coord = "x_float" if "x_float" in points.columns else "x"
    y_coord = "y_float" if "y_float" in points.columns else "y"
    return x_coord, y_coord


def _resolution_by_axis(space: SpaceDefinition) -> tuple[float, float, float]:
    """Return x/y/z resolution in microns based on axis labels."""

    resolution_map = dict(zip(space.axis_labels, space.resolution_um, strict=True))
    return (
        float(resolution_map["x"]),
        float(resolution_map["y"]),
        float(resolution_map["z"]),
    )


def _connected_components(
    detection_ids: list[int],
    detection_to_z: dict[int, int],
    edges: pd.DataFrame,
    *,
    max_n_planes: int | None = None,
) -> tuple[list[list[int]], pd.DataFrame]:
    """Find plane-unique connected components over detection IDs."""

    parent = {detection_id: detection_id for detection_id in detection_ids}
    component_planes = {detection_id: {detection_to_z[detection_id]} for detection_id in detection_ids}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
            component_planes[root_a].update(component_planes[root_b])
            del component_planes[root_b]
        else:
            parent[root_a] = root_b
            component_planes[root_b].update(component_planes[root_a])
            del component_planes[root_a]

    accepted_edge_rows: list[dict[str, int | float]] = []
    sorted_edges = edges.sort_values(
        ["xy_distance_um", "plane_offset", "source_detection_id", "target_detection_id"]
    ).reset_index(drop=True)

    for edge in sorted_edges.itertuples(index=False):
        source_detection_id = int(edge.source_detection_id)
        target_detection_id = int(edge.target_detection_id)
        root_source = find(source_detection_id)
        root_target = find(target_detection_id)

        if root_source != root_target:
            source_planes = component_planes[root_source]
            target_planes = component_planes[root_target]
            if not source_planes.isdisjoint(target_planes):
                continue
            if max_n_planes is not None:
                merged_planes = source_planes | target_planes
                if len(merged_planes) > max_n_planes:
                    continue

        union(source_detection_id, target_detection_id)
        accepted_edge_rows.append(
            {
                "source_detection_id": source_detection_id,
                "target_detection_id": target_detection_id,
                "source_z": int(edge.source_z),
                "target_z": int(edge.target_z),
                "plane_offset": int(edge.plane_offset),
                "dx_um": float(edge.dx_um),
                "dy_um": float(edge.dy_um),
                "xy_distance_um": float(edge.xy_distance_um),
                "dz_um": float(edge.dz_um),
            }
        )

    grouped: dict[int, list[int]] = {}
    for detection_id in detection_ids:
        root = find(detection_id)
        grouped.setdefault(root, []).append(detection_id)

    components = [sorted(component) for _, component in sorted(grouped.items())]
    accepted_edges = pd.DataFrame(accepted_edge_rows, columns=EDGE_COLUMNS)
    return components, accepted_edges
