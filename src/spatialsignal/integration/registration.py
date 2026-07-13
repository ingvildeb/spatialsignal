"""Helpers for consuming `atlasspace` registration output folders."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import nibabel as nib
import numpy as np

from spatialsignal.models import SpaceDefinition


_SUMMARY_PATH_KEYS = {
    "fixed_image",
    "moving_image",
    "fixed_normalized_for_registration",
    "moving_normalized_for_registration",
    "output_dir",
    "warped_image",
    "inverse_warped_image",
}


@dataclass(frozen=True)
class RegistrationOutputFolder:
    """Resolved view of an `atlasspace` registration output folder."""

    registration_dir: Path
    summary_path: Path | None
    parameters_path: Path | None
    summary: dict[str, Any]
    annotation_path: Path
    brain_mask_path: Path | None
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
    *,
    annotation_name: str = "annotation",
    brain_mask_name: str = "brain_mask",
) -> RegistrationOutputFolder:
    """Load the pieces of an `atlasspace` registration output folder needed downstream."""

    registration_dir = Path(registration_dir)
    summary_path = registration_dir / "registration_summary.txt"
    parameters_path = registration_dir / "registration_parameters.yaml"
    summary = _parse_registration_summary(summary_path) if summary_path.exists() else {}

    transformed_segmentations = _normalize_path_dict(
        summary.get("transformed_segmentations", {})
    )
    annotation_path = _resolve_segmentation_path(
        registration_dir,
        transformed_segmentations,
        annotation_name,
    )
    brain_mask_path = _resolve_segmentation_path(
        registration_dir,
        transformed_segmentations,
        brain_mask_name,
        required=False,
    )

    warped_image_path = _optional_path(summary.get("warped_image"))
    inverse_warped_image_path = _optional_path(summary.get("inverse_warped_image"))
    forward_transforms = _normalize_path_list(summary.get("forward_transforms", []))
    inverse_transforms = _normalize_path_list(summary.get("inverse_transforms", []))

    return RegistrationOutputFolder(
        registration_dir=registration_dir,
        summary_path=summary_path if summary_path.exists() else None,
        parameters_path=parameters_path if parameters_path.exists() else None,
        summary=summary,
        annotation_path=annotation_path,
        brain_mask_path=brain_mask_path,
        warped_image_path=warped_image_path,
        inverse_warped_image_path=inverse_warped_image_path,
        forward_transforms=forward_transforms,
        inverse_transforms=inverse_transforms,
        transformed_segmentations=transformed_segmentations,
    )


def load_registration_annotation_volume(
    registration: RegistrationOutputFolder,
) -> LabelVolume:
    """Load the warped annotation from a registration output folder."""

    space_name = str(
        registration.summary.get("fixed_space_name", "registered_subject_space")
    )
    return load_label_volume(registration.annotation_path, space_name=space_name)


def load_registration_brain_mask_volume(
    registration: RegistrationOutputFolder,
) -> LabelVolume | None:
    """Load the warped brain mask from a registration output folder if present."""

    if registration.brain_mask_path is None:
        return None

    space_name = str(
        registration.summary.get("fixed_space_name", "registered_subject_space")
    )
    return load_label_volume(registration.brain_mask_path, space_name=space_name)


def load_label_volume(
    path: Path,
    *,
    space_name: str,
    indexing: str = "zero_based",
) -> LabelVolume:
    """Load a NIfTI volume and derive a `SpaceDefinition` from its affine and shape."""

    image = nib.load(str(path))
    data = np.asarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D label volume at {path}, got shape {data.shape}")

    space = _space_from_nifti_image(image, data.shape, space_name=space_name, indexing=indexing)
    return LabelVolume(path=Path(path), data=data, space=space)


def _parse_registration_summary(summary_path: Path) -> dict[str, Any]:
    """Parse `registration_summary.txt` into a minimally typed dictionary."""

    summary: dict[str, Any] = {}
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        summary[key] = _parse_summary_value(key, raw_value)
    return summary


def _parse_summary_value(key: str, raw_value: str) -> Any:
    """Parse one summary value, preserving strings when literal parsing is unsafe."""

    raw_value = raw_value.strip()
    if key in _SUMMARY_PATH_KEYS:
        return raw_value

    sanitized = re.sub(
        r"(WindowsPath|PosixPath)\('([^']*)'\)",
        lambda match: repr(match.group(2)),
        raw_value,
    )
    try:
        return ast.literal_eval(sanitized)
    except (SyntaxError, ValueError):
        return raw_value


def _resolve_segmentation_path(
    registration_dir: Path,
    transformed_segmentations: dict[str, Path],
    segmentation_name: str,
    *,
    required: bool = True,
) -> Path | None:
    """Resolve a transformed segmentation path from summary metadata or standard naming."""

    if segmentation_name in transformed_segmentations:
        path = transformed_segmentations[segmentation_name]
        if path.exists():
            return path

    fallback = registration_dir / f"{segmentation_name}_WarpedSegmentation.nii.gz"
    if fallback.exists():
        return fallback

    if required:
        raise FileNotFoundError(
            f"Could not find transformed segmentation '{segmentation_name}' in {registration_dir}"
        )
    return None


def _normalize_path_list(value: Any) -> list[Path]:
    """Normalize a summary value into a list of paths."""

    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    return [Path(item) for item in value]


def _normalize_path_dict(value: Any) -> dict[str, Path]:
    """Normalize a summary value into a dictionary of paths."""

    if not value:
        return {}
    return {str(key): Path(path_value) for key, path_value in value.items()}


def _optional_path(value: Any) -> Path | None:
    """Normalize an optional summary value into a path."""

    if value in (None, "", "None"):
        return None
    return Path(value)


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
