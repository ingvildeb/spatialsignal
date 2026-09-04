"""Shared hemisphere conventions and validation for regional quantification."""

from __future__ import annotations

import numpy as np

from spatialsignal.integration.registration import LabelVolume
from spatialsignal.models import SpaceDefinition


LEFT_HEMISPHERE_ID = 1
RIGHT_HEMISPHERE_ID = 2
HEMISPHERE_NAMES = {
    LEFT_HEMISPHERE_ID: "left",
    RIGHT_HEMISPHERE_ID: "right",
}
MAX_HEMISPHERE_VALIDATION_CHUNK_VOXELS = 4_000_000


def validate_volume_sampling_space(
    data: np.ndarray,
    space: SpaceDefinition,
    reference: LabelVolume,
    *,
    volume_name: str,
) -> None:
    """Require a volume to use the annotation's exact sampling grid."""

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D {volume_name}, got shape {data.shape}")

    mismatches: list[str] = []
    if tuple(data.shape) != tuple(reference.data.shape):
        mismatches.append(
            f"data shape {tuple(data.shape)} vs {tuple(reference.data.shape)}"
        )
    if tuple(space.shape) != tuple(reference.space.shape):
        mismatches.append(
            f"declared shape {tuple(space.shape)} vs {tuple(reference.space.shape)}"
        )
    if space.orientation.lower() != reference.space.orientation.lower():
        mismatches.append(
            f"orientation {space.orientation} vs {reference.space.orientation}"
        )
    if space.indexing != reference.space.indexing:
        mismatches.append(
            f"indexing {space.indexing} vs {reference.space.indexing}"
        )
    if not np.allclose(
        np.asarray(space.resolution_um, dtype=float),
        np.asarray(reference.space.resolution_um, dtype=float),
    ):
        mismatches.append(
            "resolution_um "
            f"{tuple(space.resolution_um)} vs {tuple(reference.space.resolution_um)}"
        )

    affine = _optional_affine(space)
    reference_affine = _optional_affine(reference.space)
    if affine is not None and reference_affine is not None and not np.allclose(
        affine,
        reference_affine,
        rtol=1e-5,
        atol=1e-6,
    ):
        mismatches.append("affine_ras_mm differs")

    if mismatches:
        raise ValueError(
            f"{volume_name.capitalize()} is not on the annotation sampling grid: "
            + "; ".join(mismatches)
        )


def validate_hemisphere_volume(
    hemisphere: LabelVolume,
    annotation: LabelVolume,
    *,
    background_id: int = 0,
) -> None:
    """Validate a BrainGlobe-convention hemisphere map for an annotation."""

    validate_volume_sampling_space(
        hemisphere.data,
        hemisphere.space,
        annotation,
        volume_name="hemisphere volume",
    )
    annotation_data = np.asarray(annotation.data)
    hemisphere_data = np.asarray(hemisphere.data)
    plane_voxels = int(annotation_data.shape[1]) * int(annotation_data.shape[2])
    planes_per_chunk = max(
        1,
        MAX_HEMISPHERE_VALIDATION_CHUNK_VOXELS // plane_voxels,
    )
    invalid_values: set[int | float] = set()
    for start in range(0, int(annotation_data.shape[0]), planes_per_chunk):
        stop = min(start + planes_per_chunk, int(annotation_data.shape[0]))
        annotation_block = annotation_data[start:stop]
        hemisphere_block = hemisphere_data[start:stop]
        annotated = annotation_block != background_id
        invalid = annotated & ~np.isin(hemisphere_block, tuple(HEMISPHERE_NAMES))
        if np.any(invalid):
            invalid_values.update(np.unique(hemisphere_block[invalid]).tolist())
    if invalid_values:
        raise ValueError(
            "Every annotated voxel must have hemisphere ID 1 (left) or 2 "
            f"(right); found {sorted(invalid_values)}"
        )


def _optional_affine(space: SpaceDefinition) -> np.ndarray | None:
    """Return a validated affine array when one is declared."""

    if space.affine_ras_mm is None:
        return None
    affine = np.asarray(space.affine_ras_mm, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(
            f"affine_ras_mm must have shape (4, 4), got {affine.shape}"
        )
    return affine
