from pathlib import Path

import pytest

from spatialsignal.io import assign_slices, find_mask_files


def test_find_mask_files_uses_natural_sort(tmp_path: Path) -> None:
    (tmp_path / "masks_10.tif").touch()
    (tmp_path / "masks_2.tif").touch()
    (tmp_path / "masks_1.tif").touch()

    mask_files = find_mask_files(tmp_path)

    assert [path.name for path in mask_files] == [
        "masks_1.tif",
        "masks_2.tif",
        "masks_10.tif",
    ]


def test_find_mask_files_raises_when_no_files_match(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_mask_files(tmp_path)


def test_assign_slices_returns_sequential_indices() -> None:
    mask_files = [Path("masks_1.tif"), Path("masks_2.tif"), Path("masks_3.tif")]

    assigned = assign_slices(mask_files, slice_start=3)

    assert assigned == [
        (3, Path("masks_1.tif")),
        (4, Path("masks_2.tif")),
        (5, Path("masks_3.tif")),
    ]


def test_assign_slices_supports_zero_start() -> None:
    mask_files = [Path("masks_1.tif"), Path("masks_2.tif")]

    assigned = assign_slices(mask_files)

    assert assigned == [
        (0, Path("masks_1.tif")),
        (1, Path("masks_2.tif")),
    ]


def test_assign_slices_rejects_negative_start() -> None:
    with pytest.raises(ValueError):
        assign_slices([Path("masks_1.tif")], slice_start=-1)
