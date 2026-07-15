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
    """Merge raw detections that likely represent the same cell across planes."""

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
    """Aggregate raw detections into cleaned cross-plane objects."""

    merged = membership.merge(points, on="detection_id", how="left", validate="many_to_one")
    x_coord, y_coord = _xy_coordinate_columns(points)

    object_rows: list[dict[str, int | float]] = []
    for object_id, group in merged.groupby("object_id", sort=True):
        x_float = float(group[x_coord].mean())
        y_float = float(group[y_coord].mean())
        z_float = float(group["z"].mean())

        row: dict[str, int | float] = {
            "object_id": int(object_id),
            "x": int(np.floor(x_float + 0.5)),
            "y": int(np.floor(y_float + 0.5)),
            "z": int(np.floor(z_float + 0.5)),
            "x_float": x_float,
            "y_float": y_float,
            "z_float": z_float,
            "n_detections": int(len(group)),
            "n_planes": int(group["z"].nunique()),
            "z_min": int(group["z"].min()),
            "z_max": int(group["z"].max()),
        }

        if "area_px" in group.columns:
            row["mean_area_px"] = float(group["area_px"].mean())
            row["max_area_px"] = float(group["area_px"].max())
        for source_column, output_column in (
            ("major_axis_length_px", "mean_major_axis_length_px"),
            ("minor_axis_length_px", "mean_minor_axis_length_px"),
            ("eccentricity", "mean_eccentricity"),
        ):
            if source_column in group.columns:
                row[output_column] = float(group[source_column].mean())

        object_rows.append(row)

    objects = pd.DataFrame(object_rows)
    required = CLEANED_OBJECT_REQUIRED_COLUMNS
    optional_columns = [
        "mean_area_px",
        "max_area_px",
        "mean_major_axis_length_px",
        "mean_minor_axis_length_px",
        "mean_eccentricity",
    ]
    optional = [column for column in optional_columns if column in objects.columns]
    return objects[required + optional]


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
    """Find connected components over detection IDs using union-find."""

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

        if root_source != root_target and max_n_planes is not None:
            merged_planes = component_planes[root_source] | component_planes[root_target]
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
