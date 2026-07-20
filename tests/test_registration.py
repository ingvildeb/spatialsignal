"""Tests for loading label volumes from registration outputs or direct paths."""

from pathlib import Path

import nibabel as nib
import numpy as np

from spatialsignal.integration import load_label_volume


def test_load_label_volume_infers_space_name_from_nifti_filename(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotation_WarpedSegmentation.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint16), np.eye(4)),
        annotation_path,
    )

    annotation = load_label_volume(annotation_path)

    assert annotation.path == annotation_path
    assert annotation.space.space_name == "annotation_WarpedSegmentation"
    assert annotation.space.shape == [3, 4, 5]


def test_load_label_volume_accepts_explicit_space_name(tmp_path: Path) -> None:
    annotation_path = tmp_path / "annotation.nii"
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint16), np.eye(4)),
        annotation_path,
    )

    annotation = load_label_volume(annotation_path, space_name="subject_001")

    assert annotation.space.space_name == "subject_001"
