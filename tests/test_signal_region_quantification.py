"""Tests for atlas-aware regional semantic-signal quantification."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialsignal.integration import LabelVolume
from spatialsignal.models import SpaceDefinition
from spatialsignal.quantification import (
    SignalRegionQuantificationResult,
    quantify_signal_by_region,
    signal_regions,
)


def _space() -> SpaceDefinition:
    return SpaceDefinition(
        space_name="subject_annotation",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=[4, 1, 1],
        resolution_um=[10.0, 10.0, 10.0],
    )


def _volume(name: str, values: list[float], dtype: np.dtype) -> LabelVolume:
    return LabelVolume(
        path=Path(f"{name}.nii.gz"),
        data=np.asarray(values, dtype=dtype).reshape(4, 1, 1),
        space=_space(),
    )


def test_quantify_signal_by_region_reports_fractional_occupancy_by_side(
    tmp_path: Path,
) -> None:
    annotation = _volume("annotation", [68, 68, 667, 667], np.int32)
    hemisphere = _volume("hemispheres", [1, 2, 1, 2], np.uint8)
    signal = np.asarray([1.0, 0.5, 0.0, 1.0], dtype=np.float32).reshape(4, 1, 1)
    output_csv = tmp_path / "subject_damage_by_region.csv"

    result = quantify_signal_by_region(
        signal,
        annotation,
        hemisphere=hemisphere,
        output_csv=output_csv,
    )

    assert isinstance(result, SignalRegionQuantificationResult)
    assert result.summary_csv == output_csv
    region_68 = result.region_summary.loc[result.region_summary["region_id"] == 68].iloc[0]
    assert region_68["bilateral_region_voxels"] == 2
    assert region_68["bilateral_signal_voxel_equivalents"] == 1.5
    assert region_68["bilateral_signal_fraction"] == 0.75
    assert region_68["left_signal_fraction"] == 1.0
    assert region_68["right_signal_fraction"] == 0.5
    assert region_68["bilateral_region_volume_mm3"] == pytest.approx(2e-6)
    assert region_68["bilateral_signal_volume_mm3"] == pytest.approx(1.5e-6)
    pd.testing.assert_frame_equal(
        pd.read_csv(output_csv),
        result.region_summary,
        check_dtype=False,
    )


def test_quantify_signal_by_region_without_hemisphere_is_bilateral_only() -> None:
    annotation = _volume("annotation", [68, 68, 667, 667], np.int32)
    signal = np.asarray([1, 0, 0, 1], dtype=np.uint8).reshape(4, 1, 1)

    result = quantify_signal_by_region(signal, annotation)

    assert "bilateral_signal_fraction" in result.region_summary.columns
    assert not any(column.startswith("left_") for column in result.region_summary)
    assert not any(column.startswith("right_") for column in result.region_summary)


@pytest.mark.parametrize(
    "signal, message",
    [
        (np.asarray([0.0, -0.1, 0.0, 1.0]).reshape(4, 1, 1), "in \\[0, 1\\]"),
        (np.asarray([0.0, 1.1, 0.0, 1.0]).reshape(4, 1, 1), "in \\[0, 1\\]"),
        (np.asarray([0.0, np.nan, 0.0, 1.0]).reshape(4, 1, 1), "finite"),
    ],
)
def test_quantify_signal_by_region_rejects_invalid_signal(
    signal: np.ndarray,
    message: str,
) -> None:
    annotation = _volume("annotation", [68, 68, 667, 667], np.int32)

    with pytest.raises(ValueError, match=message):
        quantify_signal_by_region(signal, annotation)


def test_quantify_signal_by_region_rejects_invalid_hemisphere_labels() -> None:
    annotation = _volume("annotation", [68, 68, 667, 667], np.int32)
    hemisphere = _volume("hemispheres", [1, 2, 0, 2], np.uint8)
    signal = np.zeros((4, 1, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="Every annotated voxel"):
        quantify_signal_by_region(signal, annotation, hemisphere=hemisphere)


def test_quantify_signal_by_region_validates_signal_volume_grid() -> None:
    annotation = _volume("annotation", [68, 68, 667, 667], np.int32)
    mismatched_space = _space()
    mismatched_space.orientation = "rai"
    signal = LabelVolume(
        path=Path("damage.nii.gz"),
        data=np.zeros((4, 1, 1), dtype=np.uint8),
        space=mismatched_space,
    )

    with pytest.raises(ValueError, match="not on the annotation sampling grid"):
        quantify_signal_by_region(signal, annotation)


def test_chunked_aggregation_matches_direct_fractional_calculation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signal_regions, "DEFAULT_MAX_AGGREGATION_CHUNK_VOXELS", 4)
    annotation_values = np.asarray(
        [
            0, 68, 68, 667,
            667, 68, 0, 667,
            68, 667, 667, 68,
            0, 0, 68, 667,
            667, 68, 68, 667,
            68, 667, 0, 68,
        ],
        dtype=np.int32,
    ).reshape(6, 2, 2)
    signal_values = np.linspace(0.0, 1.0, annotation_values.size).reshape(
        annotation_values.shape
    )
    hemisphere_values = np.where(
        np.indices(annotation_values.shape)[0] < 3,
        1,
        2,
    ).astype(np.uint8)
    hemisphere_values[annotation_values == 0] = 0
    space = SpaceDefinition(
        space_name="chunked_test",
        orientation="las",
        axis_labels=["x", "y", "z"],
        indexing="zero_based",
        units="voxel",
        shape=list(annotation_values.shape),
        resolution_um=[10.0, 10.0, 10.0],
    )
    annotation = LabelVolume(Path("annotation.nii.gz"), annotation_values, space)
    hemisphere = LabelVolume(Path("hemispheres.nii.gz"), hemisphere_values, space)

    result = quantify_signal_by_region(
        signal_values,
        annotation,
        hemisphere=hemisphere,
    ).region_summary

    for region_id in (68, 667):
        row = result.loc[result["region_id"] == region_id].iloc[0]
        region = annotation_values == region_id
        for scope_name, scope in (
            ("bilateral", np.ones(annotation_values.shape, dtype=bool)),
            ("left", hemisphere_values == 1),
            ("right", hemisphere_values == 2),
        ):
            selected = region & scope
            expected_count = int(np.count_nonzero(selected))
            expected_sum = float(signal_values[selected].sum(dtype=np.float64))
            assert row[f"{scope_name}_region_voxels"] == expected_count
            assert row[f"{scope_name}_signal_voxel_equivalents"] == pytest.approx(
                expected_sum
            )


def test_internal_aggregation_supports_sparse_and_negative_region_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(signal_regions, "DEFAULT_MAX_AGGREGATION_CHUNK_VOXELS", 2)
    monkeypatch.setattr(signal_regions, "MAX_DENSE_REGION_LOOKUP_SIZE", 4)
    annotation = np.asarray(
        [-1, 68, 900_000_001, 68, -1, 900_000_001],
        dtype=np.int64,
    ).reshape(6, 1, 1)
    signal = np.asarray([1.0, 0.5, 0.25, 1.0, 0.0, 0.75]).reshape(6, 1, 1)
    hemisphere = np.asarray([0, 1, 2, 2, 0, 1], dtype=np.uint8).reshape(6, 1, 1)

    region_ids, counts, sums = signal_regions._aggregate_signal_by_region(
        signal,
        annotation,
        hemisphere,
    )

    assert region_ids.tolist() == [-1, 68, 900_000_001]
    np.testing.assert_array_equal(
        counts,
        np.asarray([[2, 0, 0], [2, 1, 1], [2, 1, 1]]),
    )
    np.testing.assert_allclose(
        sums,
        np.asarray([[1.0, 0.0, 0.0], [1.5, 0.5, 1.0], [1.0, 0.75, 0.25]]),
    )
