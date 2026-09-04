"""Tests for loading label volumes from registration outputs or direct paths."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from spatialsignal.integration import (
    load_atlasspace_registration_folder,
    load_label_volume,
)


def _write_registration_manifest(
    registration_dir: Path,
    *,
    success: bool = True,
) -> None:
    registration_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "success": success,
        "parameters_snapshot": "registration_parameters.yaml",
        "fixed_image": {
            "image": "Z:/source/fixed.nii.gz",
            "normalized_image": "fixed_normalized_for_registration.nii.gz",
            "space": {"space_name": "subject_001"},
        },
        "moving_image": {
            "image": "Z:/templates/moving.nii.gz",
            "normalized_image": "moving_normalized_for_registration.nii.gz",
            "space": {"space_name": "template_p56"},
        },
        "effective_fixed_space": {"space_name": "subject_001"},
        "effective_moving_space": {"space_name": "template_p56"},
        "warped_image": "warped.nii.gz",
        "inverse_warped_image": "inverse_warped.nii.gz",
        "forward_transforms": ["forward.mat", "forward_warp.nii.gz"],
        "inverse_transforms": ["inverse.mat", "inverse_warp.nii.gz"],
        "transformed_segmentations": {
            "labels": "labels_WarpedSegmentation.nii.gz",
            "hemispheres": "hemispheres_WarpedSegmentation.nii.gz",
        },
    }
    (registration_dir / "registration_result.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )


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


def test_load_registration_folder_resolves_generic_manifest_paths(
    tmp_path: Path,
) -> None:
    registration_dir = tmp_path / "registration"
    _write_registration_manifest(registration_dir)

    registration = load_atlasspace_registration_folder(registration_dir)

    assert registration.manifest_path == registration_dir / "registration_result.json"
    assert registration.parameters_path == registration_dir / "registration_parameters.yaml"
    assert registration.fixed_normalized_image_path == (
        registration_dir / "fixed_normalized_for_registration.nii.gz"
    )
    assert registration.inverse_transforms == [
        registration_dir / "inverse.mat",
        registration_dir / "inverse_warp.nii.gz",
    ]
    assert registration.transformed_segmentations == {
        "labels": registration_dir / "labels_WarpedSegmentation.nii.gz",
        "hemispheres": registration_dir / "hemispheres_WarpedSegmentation.nii.gz",
    }


def test_registration_segmentations_use_the_same_generic_volume_loader(
    tmp_path: Path,
) -> None:
    registration_dir = tmp_path / "registration"
    _write_registration_manifest(registration_dir)
    for name, value in (("labels", 68), ("hemispheres", 1)):
        nib.save(
            nib.Nifti1Image(np.full((2, 2, 2), value, dtype=np.uint16), np.eye(4)),
            registration_dir / f"{name}_WarpedSegmentation.nii.gz",
        )

    registration = load_atlasspace_registration_folder(registration_dir)
    labels = load_label_volume(registration.transformed_segmentations["labels"])
    hemispheres = load_label_volume(
        registration.transformed_segmentations["hemispheres"],
        space_name=labels.space.space_name,
    )

    assert np.all(labels.data == 68)
    assert np.all(hemispheres.data == 1)
    assert hemispheres.space.space_name == labels.space.space_name


def test_load_registration_folder_requires_successful_canonical_manifest(
    tmp_path: Path,
) -> None:
    registration_dir = tmp_path / "registration"
    _write_registration_manifest(registration_dir, success=False)

    with pytest.raises(ValueError, match="was not successful"):
        load_atlasspace_registration_folder(registration_dir)


def test_load_registration_folder_does_not_fall_back_to_legacy_summary(
    tmp_path: Path,
) -> None:
    registration_dir = tmp_path / "registration"
    registration_dir.mkdir()
    (registration_dir / "registration_summary.txt").write_text(
        "transformed_segmentations={'labels': 'labels.nii.gz'}\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="registration_result.json"):
        load_atlasspace_registration_folder(registration_dir)
