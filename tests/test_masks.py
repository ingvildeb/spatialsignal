from pathlib import Path

import pytest

from spatialsignal.io import assign_slices, find_mask_files


def test_find_mask_files_uses_natural_sort(tmp_path: Path) -> None:
    (tmp_path / "plane_10.tiff").touch()
    (tmp_path / "plane_2.tif").touch()
    (tmp_path / "plane_1.TIF").touch()

    mask_files = find_mask_files(tmp_path)

    assert [path.name for path in mask_files] == [
        "plane_1.TIF",
        "plane_2.tif",
        "plane_10.tiff",
    ]


def test_find_mask_files_filters_by_prefix_and_suffix(tmp_path: Path) -> None:
    (tmp_path / "masks_1_cellpose.tif").touch()
    (tmp_path / "masks_2_other.tif").touch()
    (tmp_path / "image_1_cellpose.tif").touch()

    mask_files = find_mask_files(tmp_path, prefix="masks_", suffix="_cellpose")

    assert [path.name for path in mask_files] == ["masks_1_cellpose.tif"]


def test_find_mask_files_supports_configurable_extensions(tmp_path: Path) -> None:
    (tmp_path / "mask_1.png").touch()
    (tmp_path / "mask_2.tif").touch()

    mask_files = find_mask_files(tmp_path, extensions=("png",))

    assert [path.name for path in mask_files] == ["mask_1.png"]


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
