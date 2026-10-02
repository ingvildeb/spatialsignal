"""Registration-aware transformation of point-cloud datasets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatialsignal.integration.registration import (
    RegistrationOutputFolder,
    load_atlasspace_registration_folder,
    load_nifti_space,
)
from spatialsignal.models import PointCloudDataset, ProcessingProvenance, SpaceDefinition
from spatialsignal.quantification import remap_pointcloud_dataset_to_space


@dataclass(frozen=True)
class ReferencePointTransformResult:
    """Reference-space points and complete rejection/accounting information."""

    dataset: PointCloudDataset
    rejected_points: pd.DataFrame
    input_count: int
    transformed_count: int
    in_bounds_count: int
    non_finite_count: int
    out_of_bounds_count: int


def _transform_registration_indices(
    source_indices_xyz: np.ndarray,
    registration_dir: Path,
    *,
    chunk_size: int | None,
) -> np.ndarray:
    try:
        from atlasspace.transforms import transform_points_from_registration
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Reference-space point transformation requires the registration extra. "
            "Install spatialsignal[registration]."
        ) from exc

    return transform_points_from_registration(
        source_indices_xyz,
        registration_dir,
        source_role="fixed",
        target_role="moving",
        chunk_size=chunk_size,
    )


def _manifest_space_name(folder: RegistrationOutputFolder, role: str) -> str:
    effective = folder.manifest.get(f"effective_{role}_space")
    if not isinstance(effective, dict):
        raise ValueError(f"Registration manifest is missing effective_{role}_space.")
    value = effective.get("space_name")
    if not isinstance(value, str) or not value.strip():
        return f"normalized_{role}_registration_space"
    return value


def _validate_manifest_geometry(
    folder: RegistrationOutputFolder,
    role: str,
    observed: SpaceDefinition,
) -> None:
    expected = folder.manifest.get(f"effective_{role}_space")
    if not isinstance(expected, dict):
        raise ValueError(f"Registration manifest is missing effective_{role}_space.")
    mismatches: list[str] = []
    if expected.get("orientation", "").lower() != observed.orientation.lower():
        mismatches.append("orientation")
    expected_shape = expected.get("shape")
    if expected_shape is not None and list(expected_shape) != list(observed.shape):
        mismatches.append("shape")
    expected_resolution = expected.get("resolution_um")
    if expected_resolution is None or not np.allclose(
        np.asarray(expected_resolution, dtype=float),
        np.asarray(observed.resolution_um, dtype=float),
    ):
        mismatches.append("resolution_um")
    if mismatches:
        raise ValueError(
            f"Normalized {role} image geometry disagrees with the registration manifest: "
            + ", ".join(mismatches)
        )


def _registration_spaces(
    folder: RegistrationOutputFolder,
) -> tuple[SpaceDefinition, SpaceDefinition]:
    if folder.fixed_normalized_image_path is None:
        raise ValueError("Registration manifest does not define a normalized fixed image.")
    if folder.moving_normalized_image_path is None:
        raise ValueError("Registration manifest does not define a normalized moving image.")
    for path, label in (
        (folder.fixed_normalized_image_path, "fixed"),
        (folder.moving_normalized_image_path, "moving"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Normalized {label} registration image is missing: {path}")

    fixed = load_nifti_space(
        folder.fixed_normalized_image_path,
        space_name=_manifest_space_name(folder, "fixed"),
    )
    moving = load_nifti_space(
        folder.moving_normalized_image_path,
        space_name=_manifest_space_name(folder, "moving"),
    )
    _validate_manifest_geometry(folder, "fixed", fixed)
    _validate_manifest_geometry(folder, "moving", moving)
    return fixed, moving


def _source_name(dataset: PointCloudDataset) -> str:
    processing = dataset.metadata.processing
    if processing is not None and processing.source_name:
        return processing.source_name
    return dataset.subject_name


def transform_pointcloud_to_reference_space(
    dataset: PointCloudDataset,
    registration_dir: Path | str,
    *,
    chunk_size: int | None = 250_000,
) -> ReferencePointTransformResult:
    """Transform native point centroids to the registration's moving/template grid."""

    dataset.validate_spatial_points()
    source_coordinate_columns = (
        ["x_float", "y_float", "z_float"]
        if {"x_float", "y_float", "z_float"} <= set(dataset.points.columns)
        else ["x", "y", "z"]
    )
    source_coordinates = dataset.points[source_coordinate_columns].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    if not np.all(np.isfinite(source_coordinates)):
        raise ValueError("Native point coordinates must contain only finite values.")
    resolved_registration_dir = Path(registration_dir)
    folder = load_atlasspace_registration_folder(resolved_registration_dir)
    fixed_space, moving_space = _registration_spaces(folder)

    fixed_dataset = remap_pointcloud_dataset_to_space(
        dataset,
        fixed_space,
        prefer_float_columns=True,
        copy_source_coordinates=True,
        source_coordinate_prefix="native",
        source_name=_source_name(dataset),
    )
    for axis in ("x", "y", "z"):
        native_float = f"native_{axis}_float"
        native_integer = f"native_{axis}"
        fixed_dataset.points[native_float] = dataset.points[
            f"{axis}_float" if f"{axis}_float" in dataset.points else axis
        ].to_numpy(copy=True)
        fixed_dataset.points[native_integer] = dataset.points[axis].to_numpy(copy=True)

    source_indices = fixed_dataset.points[["x_float", "y_float", "z_float"]].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    transformed = np.asarray(
        _transform_registration_indices(
            source_indices,
            resolved_registration_dir,
            chunk_size=chunk_size,
        ),
        dtype=np.float64,
    )
    if transformed.shape != source_indices.shape:
        raise ValueError(
            "AtlasSpace returned a transformed point array with the wrong shape: "
            f"{transformed.shape} vs {source_indices.shape}."
        )

    points = fixed_dataset.points.copy()
    points[["x_float", "y_float", "z_float"]] = transformed
    finite_mask = np.all(np.isfinite(transformed), axis=1)
    rounded = np.zeros_like(transformed, dtype=np.int64)
    rounded[finite_mask] = np.floor(transformed[finite_mask] + 0.5).astype(np.int64)
    points[["x", "y", "z"]] = rounded
    shape = np.asarray(moving_space.shape, dtype=np.int64)
    in_bounds_mask = finite_mask & np.all((rounded >= 0) & (rounded < shape), axis=1)
    non_finite_mask = ~finite_mask
    out_of_bounds_mask = finite_mask & ~in_bounds_mask

    rejected = points.loc[~in_bounds_mask].copy()
    rejected["rejection_reason"] = np.where(
        non_finite_mask[~in_bounds_mask],
        "non_finite",
        "out_of_bounds",
    )
    accepted = points.loc[in_bounds_mask].reset_index(drop=True)
    rejected = rejected.reset_index(drop=True)

    counts = {
        "input_points": int(len(points)),
        "transformed_points": int(len(transformed)),
        "in_bounds_points": int(in_bounds_mask.sum()),
        "non_finite_points": int(non_finite_mask.sum()),
        "out_of_bounds_points": int(out_of_bounds_mask.sum()),
    }
    if counts["input_points"] != (
        counts["in_bounds_points"]
        + counts["non_finite_points"]
        + counts["out_of_bounds_points"]
    ):
        raise RuntimeError("Reference point accounting invariant failed.")

    parameters: dict[str, Any] = {
        "registration_manifest": str(folder.manifest_path),
        "registration_preset": folder.manifest.get("preset_name"),
        "native_space": dataset.space.to_dict(),
        "fixed_registration_space": fixed_space.to_dict(),
        "source_role": "fixed",
        "target_role": "moving",
        "forward_transforms": [path.name for path in folder.forward_transforms],
        "inverse_transforms": [path.name for path in folder.inverse_transforms],
        "chunk_size": chunk_size,
        "modulation": "none",
    }
    metadata = replace(
        dataset.metadata,
        space=moving_space,
        processing=ProcessingProvenance(
            stage="transform_pointcloud_to_reference_space",
            source_name=_source_name(dataset),
            parameters=parameters,
            summary=counts,
        ),
    )
    transformed_dataset = PointCloudDataset(
        subject_name=dataset.subject_name,
        points=accepted,
        metadata=metadata,
    )
    transformed_dataset.validate_spatial_points()
    return ReferencePointTransformResult(
        dataset=transformed_dataset,
        rejected_points=rejected,
        input_count=counts["input_points"],
        transformed_count=counts["transformed_points"],
        in_bounds_count=counts["in_bounds_points"],
        non_finite_count=counts["non_finite_points"],
        out_of_bounds_count=counts["out_of_bounds_points"],
    )
