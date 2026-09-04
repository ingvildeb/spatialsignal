"""Helpers for consuming `atlasspace` registration output folders."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from spatialsignal.models import SpaceDefinition


REGISTRATION_RESULT_FILENAME = "registration_result.json"


@dataclass(frozen=True)
class RegistrationOutputFolder:
    """Resolved view of an `atlasspace` registration output folder."""

    registration_dir: Path
    manifest_path: Path
    parameters_path: Path | None
    manifest: dict[str, Any]
    fixed_image_path: Path
    moving_image_path: Path
    fixed_normalized_image_path: Path | None
    moving_normalized_image_path: Path | None
    warped_image_path: Path | None
    inverse_warped_image_path: Path | None
    forward_transforms: list[Path]
    inverse_transforms: list[Path]
    transformed_segmentations: dict[str, Path]


@dataclass(frozen=True)
class LabelVolume:
    """Loaded label-like NIfTI data together with interpreted spatial metadata."""

    path: Path
    data: np.ndarray
    space: SpaceDefinition


def load_atlasspace_registration_folder(
    registration_dir: Path,
) -> RegistrationOutputFolder:
    """Load an `atlasspace` result manifest and resolve every declared path.

    Transformed segmentations remain a generic mapping keyed by their registration
    segmentation IDs. Callers select the desired entry and load it with
    :func:`load_label_volume`; annotations, brain masks, and hemisphere maps do
    not require separate loader functions.
    """

    registration_dir = Path(registration_dir)
    manifest_path = registration_dir / REGISTRATION_RESULT_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"AtlasSpace registration manifest is missing: {manifest_path}"
        )
    manifest = _load_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(
            "Unsupported AtlasSpace registration manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    if manifest.get("success") is not True:
        raise ValueError(f"AtlasSpace registration was not successful: {manifest_path}")

    fixed_image = _required_mapping(manifest, "fixed_image", manifest_path)
    moving_image = _required_mapping(manifest, "moving_image", manifest_path)
    transformed_segmentations = _resolve_path_mapping(
        registration_dir,
        manifest.get("transformed_segmentations", {}),
        field_name="transformed_segmentations",
    )
    parameters_path = _resolve_optional_path(
        registration_dir,
        manifest.get("parameters_snapshot"),
        field_name="parameters_snapshot",
    )
    fixed_image_path = _resolve_required_path(
        registration_dir,
        fixed_image.get("image"),
        field_name="fixed_image.image",
    )
    moving_image_path = _resolve_required_path(
        registration_dir,
        moving_image.get("image"),
        field_name="moving_image.image",
    )

    return RegistrationOutputFolder(
        registration_dir=registration_dir,
        manifest_path=manifest_path,
        parameters_path=parameters_path,
        manifest=manifest,
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        fixed_normalized_image_path=_resolve_optional_path(
            registration_dir,
            fixed_image.get("normalized_image"),
            field_name="fixed_image.normalized_image",
        ),
        moving_normalized_image_path=_resolve_optional_path(
            registration_dir,
            moving_image.get("normalized_image"),
            field_name="moving_image.normalized_image",
        ),
        warped_image_path=_resolve_optional_path(
            registration_dir,
            manifest.get("warped_image"),
            field_name="warped_image",
        ),
        inverse_warped_image_path=_resolve_optional_path(
            registration_dir,
            manifest.get("inverse_warped_image"),
            field_name="inverse_warped_image",
        ),
        forward_transforms=_resolve_path_list(
            registration_dir,
            manifest.get("forward_transforms", []),
            field_name="forward_transforms",
        ),
        inverse_transforms=_resolve_path_list(
            registration_dir,
            manifest.get("inverse_transforms", []),
            field_name="inverse_transforms",
        ),
        transformed_segmentations=transformed_segmentations,
    )


def load_label_volume(
    path: Path,
    *,
    space_name: str | None = None,
    indexing: str = "zero_based",
) -> LabelVolume:
    """Load a NIfTI label volume and derive its spatial grid from the image.

    ``space_name`` is an optional provenance label. When omitted, it is inferred
    from the NIfTI filename without the ``.nii`` or ``.nii.gz`` suffix.
    """

    path = Path(path)
    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D label volume at {path}, got shape {data.shape}")

    resolved_space_name = space_name if space_name is not None else _nifti_stem(path)
    space = _space_from_nifti_image(
        image,
        data.shape,
        space_name=resolved_space_name,
        indexing=indexing,
    )
    return LabelVolume(path=path, data=data, space=space)


def _nifti_stem(path: Path) -> str:
    """Return a filename without a single or compressed NIfTI suffix."""

    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON object with a concise schema error."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in AtlasSpace manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"AtlasSpace manifest must contain a JSON object: {path}")
    return value


def _required_mapping(
    manifest: dict[str, Any],
    field_name: str,
    manifest_path: Path,
) -> dict[str, Any]:
    """Read a required object-valued manifest field."""

    value = manifest.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(
            f"AtlasSpace manifest field {field_name!r} must be an object: "
            f"{manifest_path}"
        )
    return value


def _resolve_required_path(
    registration_dir: Path,
    value: Any,
    *,
    field_name: str,
) -> Path:
    """Resolve a required manifest path relative to its registration folder."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"AtlasSpace manifest field {field_name!r} must be a non-empty path"
        )
    return _resolve_path(registration_dir, value)


def _resolve_optional_path(
    registration_dir: Path,
    value: Any,
    *,
    field_name: str,
) -> Path | None:
    """Resolve an optional manifest path relative to its registration folder."""

    if value is None:
        return None
    return _resolve_required_path(
        registration_dir,
        value,
        field_name=field_name,
    )


def _resolve_path_list(
    registration_dir: Path,
    value: Any,
    *,
    field_name: str,
) -> list[Path]:
    """Resolve one list of manifest paths."""

    if not isinstance(value, list):
        raise ValueError(f"AtlasSpace manifest field {field_name!r} must be a list")
    return [
        _resolve_required_path(
            registration_dir,
            item,
            field_name=f"{field_name}[{index}]",
        )
        for index, item in enumerate(value)
    ]


def _resolve_path_mapping(
    registration_dir: Path,
    value: Any,
    *,
    field_name: str,
) -> dict[str, Path]:
    """Resolve one string-keyed mapping of manifest paths."""

    if not isinstance(value, dict):
        raise ValueError(f"AtlasSpace manifest field {field_name!r} must be an object")
    resolved: dict[str, Path] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"AtlasSpace manifest field {field_name!r} has an invalid key: {key!r}"
            )
        resolved[key] = _resolve_required_path(
            registration_dir,
            item,
            field_name=f"{field_name}.{key}",
        )
    return resolved


def _resolve_path(registration_dir: Path, value: str) -> Path:
    """Resolve one absolute or registration-relative path."""

    path = Path(value)
    return path if path.is_absolute() else registration_dir / path


def _space_from_nifti_image(
    image: nib.Nifti1Image,
    shape: tuple[int, int, int],
    *,
    space_name: str,
    indexing: str,
) -> SpaceDefinition:
    """Derive a `SpaceDefinition` from a NIfTI image affine and shape."""

    axis_codes = nib.orientations.aff2axcodes(image.affine)
    origin_letter_map = {
        "R": "l",
        "L": "r",
        "A": "p",
        "P": "a",
        "S": "i",
        "I": "s",
    }
    orientation = "".join(origin_letter_map[code] for code in axis_codes)
    resolution_um = _header_zooms_to_um(image.header)

    return SpaceDefinition(
        space_name=space_name,
        orientation=orientation,
        axis_labels=["x", "y", "z"],
        indexing=indexing,
        units="voxel",
        shape=[int(value) for value in shape],
        resolution_um=resolution_um,
        affine_ras_mm=np.asarray(image.affine, dtype=np.float64).tolist(),
    )


def _header_zooms_to_um(header: nib.Nifti1Header) -> list[float]:
    """Convert NIfTI header voxel sizes to micrometers for internal metadata."""

    zooms = [float(value) for value in header.get_zooms()[:3]]
    spatial_unit = (header.get_xyzt_units()[0] or "unknown").lower()

    if spatial_unit in {"meter", "m"}:
        scale_to_um = 1_000_000.0
    elif spatial_unit in {"mm", "millimeter", "millimeters"}:
        scale_to_um = 1_000.0
    elif spatial_unit in {"micron", "micrometer", "micrometers", "um"}:
        scale_to_um = 1.0
    elif spatial_unit == "unknown":
        # NIfTI affines are conventionally interpreted in millimeters, and nibabel
        # defaults to millimeter geometry when writing standard images.
        scale_to_um = 1_000.0
    else:
        raise ValueError(f"Unsupported NIfTI spatial unit: {spatial_unit}")

    return [value * scale_to_um for value in zooms]
