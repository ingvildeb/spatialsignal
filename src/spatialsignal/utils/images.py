"""Shared image-reading helpers."""

from pathlib import Path

import numpy as np
from skimage.io import imread


def read_2d_mask(path: Path) -> np.ndarray:
    """Read a PNG or TIFF mask and require a two-dimensional image."""

    mask = np.asarray(imread(path))
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask at {path}, got shape {mask.shape}")
    return mask
