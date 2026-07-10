"""NIfTI-specific affine helpers for voxel-map export."""

from __future__ import annotations

import numpy as np

from spatialsignal.models import SpaceDefinition


def build_nifti_ras_affine(space: SpaceDefinition) -> np.ndarray:
    """Build a NIfTI-style RAS affine from BrainGlobe origin-based orientation.

    ``SpaceDefinition.orientation`` uses the BrainGlobe convention: each letter
    describes the anatomical side at voxel index 0 for the corresponding axis.
    This helper converts that origin-based convention into NIfTI's affine-based
    axis-direction convention in an RAS+ world.
    """

    if len(space.orientation) != 3:
        raise ValueError(
            f"orientation must have length 3, got {space.orientation}"
        )

    origin_letter_to_world = {
        "l": (0, 1.0),
        "r": (0, -1.0),
        "p": (1, 1.0),
        "a": (1, -1.0),
        "i": (2, 1.0),
        "s": (2, -1.0),
    }

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = 0.0
    for axis_index, (orient_letter, resolution_um) in enumerate(
        zip(
            space.orientation.lower(),
            space.resolution_um,
            strict=True,
        )
    ):
        if orient_letter not in origin_letter_to_world:
            raise ValueError(
                f"Unsupported orientation letter {orient_letter} in {space.orientation}"
            )
        world_axis, sign = origin_letter_to_world[orient_letter]
        affine[world_axis, axis_index] = sign * float(resolution_um)

    return affine
