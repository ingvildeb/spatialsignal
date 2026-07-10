"""Helpers for deriving and validating voxelization spaces."""

from __future__ import annotations

import numpy as np

from lsfm_cell_mapping.models import SpaceDefinition


def make_subject_analysis_space(
    source_space: SpaceDefinition,
    analysis_resolution_um: list[float],
    *,
    space_name: str = "subject_analysis_space",
) -> SpaceDefinition:
    """Derive a downsampled subject analysis space from a native source space."""

    if len(analysis_resolution_um) != 3:
        raise ValueError(
            "analysis_resolution_um must have length 3, got "
            f"{analysis_resolution_um}"
        )

    target_resolution = np.asarray(analysis_resolution_um, dtype=np.float64)
    if (target_resolution <= 0).any():
        raise ValueError(
            "analysis_resolution_um values must all be positive, got "
            f"{analysis_resolution_um}"
        )

    source_shape = np.asarray(source_space.shape, dtype=np.int64)
    source_resolution = np.asarray(source_space.resolution_um, dtype=np.float64)
    physical_extent_um = source_shape * source_resolution
    target_shape = np.ceil(physical_extent_um / target_resolution).astype(np.int64)

    return SpaceDefinition(
        space_name=space_name,
        orientation=source_space.orientation,
        axis_labels=list(source_space.axis_labels),
        indexing=source_space.indexing,
        units=source_space.units,
        shape=[int(value) for value in target_shape],
        resolution_um=[float(value) for value in target_resolution],
    )
