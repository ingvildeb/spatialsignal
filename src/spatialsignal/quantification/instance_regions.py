"""Subject-space region assignment and summarization for instance data."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from spatialsignal.models import PointCloudDataset, ProcessingProvenance, SpaceDefinition


OUT_OF_BOUNDS_REGION_ID = -1
REGION_ASSIGNMENT_COLUMNS = ["x", "y", "z"]
FLOAT_REGION_ASSIGNMENT_COLUMNS = ["x_float", "y_float", "z_float"]


def validate_subject_space_match(
    object_space: SpaceDefinition,
    annotation_space: SpaceDefinition,
) -> None:
    """Validate that objects can be remapped into a subject-space annotation grid."""

    source_axes = _orientation_axes(object_space)
    target_axes = _orientation_axes(annotation_space)
    for target_axis, (_, target_family) in enumerate(target_axes):
        matching_source_axes = [
            source_axis
            for source_axis, (_, source_family) in enumerate(source_axes)
            if source_family == target_family
        ]
        if len(matching_source_axes) != 1:
            raise ValueError(
                "Could not determine a unique anatomical-axis mapping between object and "
                f"annotation spaces: {object_space.orientation} vs {annotation_space.orientation}"
            )

        source_axis = matching_source_axes[0]
        source_extent_um = (
            float(object_space.shape[source_axis]) * float(object_space.resolution_um[source_axis])
        )
        target_extent_um = (
            float(annotation_space.shape[target_axis])
            * float(annotation_space.resolution_um[target_axis])
        )
        tolerance_um = max(
            float(object_space.resolution_um[source_axis]),
            float(annotation_space.resolution_um[target_axis]),
        )
        if abs(source_extent_um - target_extent_um) > tolerance_um:
            raise ValueError(
                "Object space extent does not match annotation extent for anatomical axis "
                f"{target_family}: {source_extent_um} um vs {target_extent_um} um"
            )


def remap_objects_to_space(
    objects: pd.DataFrame,
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
    *,
    prefer_float_columns: bool = True,
    copy_source_coordinates: bool = True,
    source_coordinate_prefix: str = "source",
) -> pd.DataFrame:
    """Remap object coordinates from one compatible subject grid into another."""

    _validate_object_table(objects)
    validate_subject_space_match(source_space, target_space)

    target_zero_based_float_coords = _mapped_zero_based_float_coordinates(
        objects,
        source_space,
        target_space,
        prefer_float_columns=prefer_float_columns,
    )
    target_zero_based_indices = _round_zero_based_float_coordinates(
        target_zero_based_float_coords
    )
    target_offset = _coordinate_offset(target_space)

    remapped = objects.copy()
    if copy_source_coordinates:
        for column in REGION_ASSIGNMENT_COLUMNS + FLOAT_REGION_ASSIGNMENT_COLUMNS:
            if column not in remapped.columns:
                continue
            source_column = f"{source_coordinate_prefix}_{column}"
            if source_column not in remapped.columns:
                remapped[source_column] = remapped[column]

    remapped["x"] = target_zero_based_indices[:, 0] + target_offset
    remapped["y"] = target_zero_based_indices[:, 1] + target_offset
    remapped["z"] = target_zero_based_indices[:, 2] + target_offset
    remapped["x_float"] = target_zero_based_float_coords[:, 0] + float(target_offset)
    remapped["y_float"] = target_zero_based_float_coords[:, 1] + float(target_offset)
    remapped["z_float"] = target_zero_based_float_coords[:, 2] + float(target_offset)
    return remapped


def remap_pointcloud_dataset_to_space(
    dataset: PointCloudDataset,
    target_space: SpaceDefinition,
    *,
    prefer_float_columns: bool = True,
    copy_source_coordinates: bool = True,
    source_coordinate_prefix: str = "source",
    source_name: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> PointCloudDataset:
    """Remap a point-cloud dataset into another compatible subject grid."""

    dataset.validate_spatial_points()
    remapped_points = remap_objects_to_space(
        dataset.points,
        dataset.space,
        target_space,
        prefer_float_columns=prefer_float_columns,
        copy_source_coordinates=copy_source_coordinates,
        source_coordinate_prefix=source_coordinate_prefix,
    )

    remap_parameters = dict(parameters or {})
    remap_parameters.setdefault("source_space_name", dataset.space.space_name)
    remap_parameters.setdefault("source_orientation", dataset.space.orientation)
    remap_parameters.setdefault("source_resolution_um", list(dataset.space.resolution_um))
    remap_parameters.setdefault("source_shape", list(dataset.space.shape))
    remap_parameters.setdefault("target_space_name", target_space.space_name)
    remap_parameters.setdefault("target_orientation", target_space.orientation)
    remap_parameters.setdefault("target_resolution_um", list(target_space.resolution_um))
    remap_parameters.setdefault("target_shape", list(target_space.shape))

    if source_name is None and dataset.metadata.processing is not None:
        source_name = dataset.metadata.processing.source_name

    remapped_metadata = replace(
        dataset.metadata,
        space=target_space,
        processing=ProcessingProvenance(
            stage="remap_objects_to_space",
            source_name=source_name,
            parameters=remap_parameters or None,
            summary={
                "n_points": int(len(remapped_points)),
                "source_space_name": dataset.space.space_name,
                "target_space_name": target_space.space_name,
            },
        ),
    )
    remapped_dataset = PointCloudDataset(
        subject_name=dataset.subject_name,
        points=remapped_points,
        metadata=remapped_metadata,
    )
    remapped_dataset.validate_spatial_points()
    return remapped_dataset


def assign_objects_to_regions(
    objects: pd.DataFrame,
    object_space: SpaceDefinition,
    annotation_data: np.ndarray,
    *,
    annotation_space: SpaceDefinition | None = None,
    brain_mask_data: np.ndarray | None = None,
    prefer_float_columns: bool = True,
    background_id: int = 0,
    out_of_bounds_id: int = OUT_OF_BOUNDS_REGION_ID,
) -> pd.DataFrame:
    """Assign one region ID to each object by sampling a subject-space annotation volume."""

    _validate_object_table(objects)
    volume_space = annotation_space if annotation_space is not None else object_space
    if annotation_space is not None:
        _validate_same_sampling_space(
            object_space,
            annotation_space,
            volume_name="annotation",
        )
    _validate_volume_shape(annotation_data, volume_space, volume_name="annotation")
    if brain_mask_data is not None:
        _validate_volume_shape(brain_mask_data, volume_space, volume_name="brain mask")

    x_idx, y_idx, z_idx = _rounded_zero_based_indices(
        objects,
        object_space,
        prefer_float_columns=prefer_float_columns,
    )
    in_bounds = _in_bounds_mask(x_idx, y_idx, z_idx, volume_space)

    region_ids = np.full(len(objects), out_of_bounds_id, dtype=np.int64)
    region_ids[in_bounds] = np.asarray(
        annotation_data[x_idx[in_bounds], y_idx[in_bounds], z_idx[in_bounds]]
    ).astype(np.int64, copy=False)

    if brain_mask_data is not None:
        in_brain = np.zeros(len(objects), dtype=bool)
        in_brain[in_bounds] = (
            np.asarray(brain_mask_data[x_idx[in_bounds], y_idx[in_bounds], z_idx[in_bounds]]) > 0
        )
    else:
        in_brain = region_ids != background_id

    assignment_status = np.full(len(objects), "out_of_bounds", dtype=object)
    background_mask = in_bounds & (region_ids == background_id)
    assigned_mask = in_bounds & (region_ids != background_id)
    assignment_status[background_mask] = "background"
    assignment_status[assigned_mask] = "assigned"

    assigned = objects.copy()
    assigned["region_id"] = region_ids
    assigned["in_bounds"] = in_bounds
    assigned["in_brain"] = in_brain
    assigned["assignment_status"] = assignment_status
    return assigned


def summarize_objects_by_region(
    assigned_objects: pd.DataFrame,
    object_space: SpaceDefinition,
    annotation_data: np.ndarray,
    *,
    annotation_space: SpaceDefinition | None = None,
    area_measurement_space: SpaceDefinition | None = None,
    region_column: str = "region_id",
    background_id: int = 0,
    out_of_bounds_id: int = OUT_OF_BOUNDS_REGION_ID,
    include_background: bool = False,
    include_out_of_bounds: bool = True,
) -> pd.DataFrame:
    """Summarize per-region object counts, densities, and morphology."""

    if region_column not in assigned_objects.columns:
        raise ValueError(f"Assigned object table is missing '{region_column}'")
    volume_space = annotation_space if annotation_space is not None else object_space
    if annotation_space is not None:
        _validate_same_sampling_space(
            object_space,
            annotation_space,
            volume_name="annotation",
        )
    _validate_volume_shape(annotation_data, volume_space, volume_name="annotation")

    voxel_volume_um3 = float(np.prod(volume_space.resolution_um))
    region_ids, region_voxels = np.unique(annotation_data, return_counts=True)
    area_space = area_measurement_space if area_measurement_space is not None else object_space
    region_voxel_counts = {
        int(region_id): int(voxel_count)
        for region_id, voxel_count in zip(region_ids, region_voxels, strict=True)
    }
    return _summarize_objects_from_region_counts(
        assigned_objects,
        assigned_objects[region_column],
        region_voxel_counts,
        volume_space=volume_space,
        area_measurement_space=area_space,
        background_id=background_id,
        out_of_bounds_id=out_of_bounds_id,
        include_background=include_background,
        include_out_of_bounds=include_out_of_bounds,
    )


def _summarize_objects_from_region_counts(
    assigned_objects: pd.DataFrame,
    grouped_region_ids: pd.Series,
    region_voxel_counts: Mapping[int, int],
    *,
    volume_space: SpaceDefinition,
    area_measurement_space: SpaceDefinition,
    background_id: int,
    out_of_bounds_id: int,
    include_background: bool,
    include_out_of_bounds: bool,
) -> pd.DataFrame:
    """Build one region report from object groups and precomputed voxel counts."""

    if len(grouped_region_ids) != len(assigned_objects):
        raise ValueError("Grouped region IDs must contain one value per assigned object")
    if grouped_region_ids.isna().any():
        raise ValueError("Grouped region IDs contain missing values")

    grouped = assigned_objects.groupby(grouped_region_ids, sort=True)
    voxel_volume_um3 = float(np.prod(volume_space.resolution_um))
    pixel_area_um2 = _in_plane_pixel_area_um2(area_measurement_space)

    rows: list[dict[str, Any]] = []
    for region_id, voxel_count in sorted(region_voxel_counts.items()):
        region_id = int(region_id)
        if region_id == background_id and not include_background:
            continue

        group_positions = grouped.indices.get(region_id)
        group = (
            assigned_objects.iloc[group_positions]
            if group_positions is not None
            else assigned_objects.iloc[0:0]
        )
        volume_um3 = float(voxel_count) * voxel_volume_um3
        volume_mm3 = volume_um3 / 1_000_000_000.0
        object_count = int(len(group))
        row: dict[str, Any] = {
            "region_id": region_id,
            "region_voxels": int(voxel_count),
            "region_volume_mm3": volume_mm3,
            "object_count": object_count,
            "object_density_per_mm3": (object_count / volume_mm3) if volume_mm3 > 0 else np.nan,
        }

        if "n_detections" in group.columns:
            detection_count = int(group["n_detections"].sum())
            row["detection_count"] = detection_count
        if "n_planes" in group.columns and object_count > 0:
            row["mean_n_planes"] = float(group["n_planes"].mean())
        if "mean_area_px" in group.columns and object_count > 0:
            object_areas_um2 = group["mean_area_px"].astype(float) * pixel_area_um2
            row["median_object_area_um2"] = float(object_areas_um2.median())
        if "mean_eccentricity" in group.columns and object_count > 0:
            eccentricities = group["mean_eccentricity"].astype(float)
            row["median_object_eccentricity"] = float(eccentricities.median())

        rows.append(row)

    if include_out_of_bounds:
        out_of_bounds = assigned_objects.loc[grouped_region_ids == out_of_bounds_id]
        if not out_of_bounds.empty:
            row = {
                "region_id": int(out_of_bounds_id),
                "region_voxels": 0,
                "region_volume_mm3": 0.0,
                "object_count": int(len(out_of_bounds)),
                "object_density_per_mm3": np.nan,
            }
            if "n_detections" in out_of_bounds.columns:
                row["detection_count"] = int(out_of_bounds["n_detections"].sum())
            rows.append(row)

    return pd.DataFrame(rows).sort_values("region_id").reset_index(drop=True)


def _in_plane_pixel_area_um2(space: SpaceDefinition) -> float:
    """Return the physical area represented by one x-y mask pixel."""

    resolution_by_axis = dict(zip(space.axis_labels, space.resolution_um, strict=True))
    missing_axes = [axis for axis in ("x", "y") if axis not in resolution_by_axis]
    if missing_axes:
        raise ValueError(
            "Area measurement space must define x and y axes; "
            f"missing {missing_axes} from {space.axis_labels}"
        )
    return float(resolution_by_axis["x"]) * float(resolution_by_axis["y"])


def _validate_object_table(objects: pd.DataFrame) -> None:
    """Check that the object table contains the required spatial columns."""

    missing = [column for column in REGION_ASSIGNMENT_COLUMNS if column not in objects.columns]
    if missing:
        raise ValueError(f"Object table is missing required columns: {missing}")


def _validate_volume_shape(
    data: np.ndarray,
    volume_space: SpaceDefinition,
    *,
    volume_name: str,
) -> None:
    """Check that a subject-space volume matches the declared object-space shape."""

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D {volume_name} volume, got shape {data.shape}")
    if list(data.shape) != list(volume_space.shape):
        raise ValueError(
            f"{volume_name.capitalize()} shape does not match declared space shape: "
            f"{tuple(data.shape)} vs {tuple(volume_space.shape)}"
        )


def _validate_same_sampling_space(
    object_space: SpaceDefinition,
    volume_space: SpaceDefinition,
    *,
    volume_name: str,
) -> None:
    """Require object coordinates to already be expressed in the sampling grid being used."""

    mismatches: list[str] = []
    if list(object_space.shape) != list(volume_space.shape):
        mismatches.append(f"shape {tuple(object_space.shape)} vs {tuple(volume_space.shape)}")
    if object_space.orientation.lower() != volume_space.orientation.lower():
        mismatches.append(
            f"orientation {object_space.orientation} vs {volume_space.orientation}"
        )
    if object_space.indexing != volume_space.indexing:
        mismatches.append(f"indexing {object_space.indexing} vs {volume_space.indexing}")
    if not np.allclose(
        np.asarray(object_space.resolution_um, dtype=float),
        np.asarray(volume_space.resolution_um, dtype=float),
    ):
        mismatches.append(
            "resolution_um "
            f"{tuple(object_space.resolution_um)} vs {tuple(volume_space.resolution_um)}"
        )

    if mismatches:
        raise ValueError(
            "Object coordinates are not expressed in the same sampling space as the "
            f"{volume_name}. Explicitly remap them first with remap_objects_to_space() "
            "or remap_pointcloud_dataset_to_space(). "
            + "; ".join(mismatches)
        )


def _rounded_zero_based_indices(
    objects: pd.DataFrame,
    object_space: SpaceDefinition,
    *,
    prefer_float_columns: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rounded zero-based sample indices for the object table."""

    zero_based_coords = _zero_based_float_coordinates(
        objects,
        object_space,
        prefer_float_columns=prefer_float_columns,
    )
    rounded = _round_zero_based_float_coordinates(zero_based_coords)
    x = rounded[:, 0]
    y = rounded[:, 1]
    z = rounded[:, 2]
    return x, y, z


def _in_bounds_mask(
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    z_idx: np.ndarray,
    object_space: SpaceDefinition,
) -> np.ndarray:
    """Check whether rounded sample indices fall within the declared object space."""

    x_max, y_max, z_max = (int(value) for value in object_space.shape)
    return (
        (x_idx >= 0)
        & (x_idx < x_max)
        & (y_idx >= 0)
        & (y_idx < y_max)
        & (z_idx >= 0)
        & (z_idx < z_max)
    )


def _mapped_zero_based_float_coordinates(
    objects: pd.DataFrame,
    source_space: SpaceDefinition,
    target_space: SpaceDefinition,
    *,
    prefer_float_columns: bool,
) -> np.ndarray:
    """Map object coordinates between compatible subject spaces."""

    source_coords = _zero_based_float_coordinates(
        objects,
        source_space,
        prefer_float_columns=prefer_float_columns,
    )
    source_axes = _orientation_axes(source_space)
    target_axes = _orientation_axes(target_space)
    target_coords = np.zeros_like(source_coords, dtype=np.float64)

    for target_axis, (target_letter, target_family) in enumerate(target_axes):
        source_axis = next(
            source_index
            for source_index, (_, source_family) in enumerate(source_axes)
            if source_family == target_family
        )
        source_letter, _ = source_axes[source_axis]
        source_distance_um = (
            (source_coords[:, source_axis] + 0.5) * float(source_space.resolution_um[source_axis])
        )
        if source_letter == target_letter:
            target_distance_um = source_distance_um
        else:
            source_extent_um = (
                float(source_space.shape[source_axis])
                * float(source_space.resolution_um[source_axis])
            )
            target_distance_um = source_extent_um - source_distance_um

        target_coords[:, target_axis] = (
            target_distance_um / float(target_space.resolution_um[target_axis])
        ) - 0.5

    return target_coords


def _zero_based_float_coordinates(
    objects: pd.DataFrame,
    object_space: SpaceDefinition,
    *,
    prefer_float_columns: bool,
) -> np.ndarray:
    """Return object coordinates as zero-based floating-point voxel indices."""

    if prefer_float_columns and set(FLOAT_REGION_ASSIGNMENT_COLUMNS) <= set(objects.columns):
        coords = np.column_stack(
            (
                objects["x_float"].to_numpy(dtype=float),
                objects["y_float"].to_numpy(dtype=float),
                objects["z_float"].to_numpy(dtype=float),
            )
        )
    else:
        coords = np.column_stack(
            (
                objects["x"].to_numpy(dtype=float),
                objects["y"].to_numpy(dtype=float),
                objects["z"].to_numpy(dtype=float),
            )
        )

    if object_space.indexing == "one_based":
        coords -= 1.0
    return coords


def _round_zero_based_float_coordinates(coords: np.ndarray) -> np.ndarray:
    """Round zero-based float coordinates to zero-based integer voxel indices."""

    return np.floor(coords + 0.5).astype(np.int64)


def _coordinate_offset(space: SpaceDefinition) -> int:
    """Return the integer coordinate offset implied by the space indexing."""

    if space.indexing == "zero_based":
        return 0
    if space.indexing == "one_based":
        return 1
    raise ValueError(f"Unsupported indexing mode: {space.indexing}")


def _orientation_axes(space: SpaceDefinition) -> list[tuple[str, str]]:
    """Return orientation letters paired with anatomical-axis families."""

    orientation = space.orientation.strip().lower()
    return [(letter, _letter_to_family(letter)) for letter in orientation]


def _letter_to_family(letter: str) -> str:
    """Map one orientation letter to its anatomical-axis family."""

    if letter in {"l", "r"}:
        return "lr"
    if letter in {"a", "p"}:
        return "ap"
    if letter in {"s", "i"}:
        return "si"
    raise ValueError(f"Unsupported orientation letter: {letter}")
