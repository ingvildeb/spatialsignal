"""Atlas-aware regional quantification for semantic voxel signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spatialsignal.integration.registration import LabelVolume

from .atlas_regions import enrich_region_summary_with_atlas
from .hemispheres import (
    LEFT_HEMISPHERE_ID,
    RIGHT_HEMISPHERE_ID,
    validate_hemisphere_volume,
    validate_volume_sampling_space,
)


DEFAULT_MAX_AGGREGATION_CHUNK_VOXELS = 4_000_000
MAX_DENSE_REGION_LOOKUP_SIZE = 1_000_000


@dataclass(frozen=True)
class SignalRegionQuantificationResult:
    """Regional summary produced from one subject-space signal map."""

    region_summary: pd.DataFrame
    summary_csv: Path | None = None


def quantify_signal_by_region(
    signal: np.ndarray | LabelVolume,
    annotation: LabelVolume,
    *,
    hemisphere: LabelVolume | None = None,
    ontology_preset: str = "allen_ccfv3",
    region_id_space: str = "allen",
    include_background: bool = False,
    background_id: int = 0,
    output_csv: Path | None = None,
) -> SignalRegionQuantificationResult:
    """Summarize a fractional semantic signal for every atlas region.

    ``signal`` must contain finite values in ``[0, 1]``. Pass a ``LabelVolume``
    when NIfTI grid metadata is available so that exact sampling-space
    compatibility is validated; a bare array is checked against the annotation
    shape. Binary masks are the common special case. Fractional values are
    interpreted as partial voxel occupancy, so their sum is reported as
    ``signal_voxel_equivalents`` and is converted into a physical signal volume.

    When ``hemisphere`` is provided, the report contains explicit bilateral,
    left, and right columns. Hemisphere IDs follow the BrainGlobe convention:
    ``1 = left`` and ``2 = right``.
    """

    signal_data = _validate_signal(signal, annotation)
    if hemisphere is not None:
        validate_hemisphere_volume(
            hemisphere,
            annotation,
            background_id=background_id,
        )

    hemisphere_data = None
    if hemisphere is not None:
        hemisphere_data = np.asarray(hemisphere.data)

    scopes = ["bilateral"]
    if hemisphere_data is not None:
        scopes.extend(["left", "right"])
    region_ids, region_counts, signal_sums = _aggregate_signal_by_region(
        signal_data,
        np.asarray(annotation.data),
        hemisphere_data,
    )
    rows: list[dict[str, float | int]] = []
    voxel_volume_mm3 = float(np.prod(annotation.space.resolution_um)) / 1_000_000_000.0
    for region_index, region_id in enumerate(region_ids.tolist()):
        if region_id == background_id and not include_background:
            continue

        row: dict[str, float | int] = {"region_id": region_id}
        for scope_index, scope_name in enumerate(scopes):
            region_voxels = int(region_counts[region_index, scope_index])
            signal_sum = float(signal_sums[region_index, scope_index])
            row[f"{scope_name}_region_voxels"] = region_voxels
            row[f"{scope_name}_region_volume_mm3"] = (
                region_voxels * voxel_volume_mm3
            )
            row[f"{scope_name}_signal_voxel_equivalents"] = signal_sum
            row[f"{scope_name}_signal_volume_mm3"] = signal_sum * voxel_volume_mm3
            row[f"{scope_name}_signal_fraction"] = (
                signal_sum / region_voxels if region_voxels else np.nan
            )
        rows.append(row)

    region_summary = pd.DataFrame(rows)
    region_summary = enrich_region_summary_with_atlas(
        region_summary,
        ontology_preset=ontology_preset,
        region_id_space=region_id_space,
        background_id=background_id,
    )

    summary_csv = Path(output_csv) if output_csv is not None else None
    if summary_csv is not None:
        if summary_csv.suffix.lower() != ".csv":
            raise ValueError(f"output_csv must have a .csv suffix, got {summary_csv}")
        summary_csv.parent.mkdir(parents=True, exist_ok=True)
        region_summary.to_csv(summary_csv, index=False)

    return SignalRegionQuantificationResult(
        region_summary=region_summary,
        summary_csv=summary_csv,
    )


def _validate_signal(
    signal: np.ndarray | LabelVolume,
    annotation: LabelVolume,
) -> np.ndarray:
    """Validate and normalize one semantic signal array."""

    if isinstance(signal, LabelVolume):
        validate_volume_sampling_space(
            signal.data,
            signal.space,
            annotation,
            volume_name="signal map",
        )
        signal_data = np.asarray(signal.data)
    else:
        signal_data = np.asarray(signal)
    if signal_data.ndim != 3:
        raise ValueError(f"Expected a 3D signal map, got shape {signal_data.shape}")
    if signal_data.shape != annotation.data.shape:
        raise ValueError(
            "Signal shape does not match annotation shape: "
            f"{signal_data.shape} vs {annotation.data.shape}"
        )
    if not np.issubdtype(signal_data.dtype, np.number) and signal_data.dtype != bool:
        raise TypeError(f"Signal map must be numeric or boolean, got {signal_data.dtype}")
    for block_slice in _axis0_chunk_slices(
        signal_data.shape,
        max_chunk_voxels=DEFAULT_MAX_AGGREGATION_CHUNK_VOXELS,
    ):
        block = signal_data[block_slice]
        if not np.all(np.isfinite(block)):
            raise ValueError("Signal map must contain only finite values")
        if np.any(block < 0) or np.any(block > 1):
            raise ValueError("Signal map values must lie in [0, 1]")
    return signal_data


def _axis0_chunk_slices(
    shape: tuple[int, ...],
    *,
    max_chunk_voxels: int,
) -> list[tuple[slice, slice, slice]]:
    """Partition a 3D array into bounded logical blocks along its first axis."""

    if len(shape) != 3 or any(int(value) <= 0 for value in shape):
        raise ValueError(f"Expected a nonempty 3D shape, got {shape}")
    if max_chunk_voxels < 1:
        raise ValueError("max_chunk_voxels must be positive")
    plane_voxels = int(shape[1]) * int(shape[2])
    planes_per_chunk = max(1, int(max_chunk_voxels) // plane_voxels)
    return [
        (slice(start, min(start + planes_per_chunk, int(shape[0]))), slice(None), slice(None))
        for start in range(0, int(shape[0]), planes_per_chunk)
    ]


def _validate_and_collect_region_ids(annotation_data: np.ndarray) -> np.ndarray:
    """Validate integer labels and collect sorted IDs without a full-size sort."""

    if annotation_data.ndim != 3:
        raise ValueError(f"Expected a 3D annotation, got shape {annotation_data.shape}")
    if not np.issubdtype(annotation_data.dtype, np.number):
        raise TypeError(f"Annotation must be numeric, got {annotation_data.dtype}")
    region_ids: set[int] = set()
    for block_slice in _axis0_chunk_slices(
        annotation_data.shape,
        max_chunk_voxels=DEFAULT_MAX_AGGREGATION_CHUNK_VOXELS,
    ):
        block = np.asarray(annotation_data[block_slice])
        if not np.all(np.isfinite(block)):
            raise ValueError("Annotation must contain only finite values")
        if not np.issubdtype(block.dtype, np.integer):
            rounded = np.rint(block)
            if not np.array_equal(block, rounded):
                raise ValueError("Annotation must contain integer label values")
            block = rounded
        region_ids.update(int(value) for value in np.unique(block).tolist())
    return np.asarray(sorted(region_ids), dtype=np.int64)


def _build_region_indexer(region_ids: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Build a compact dense lookup when the label range makes that efficient."""

    minimum = int(region_ids[0])
    maximum = int(region_ids[-1])
    span = maximum - minimum + 1
    if span > MAX_DENSE_REGION_LOOKUP_SIZE:
        return None, minimum
    lookup = np.full(span, -1, dtype=np.int32)
    lookup[region_ids - minimum] = np.arange(len(region_ids), dtype=np.int32)
    return lookup, minimum


def _region_indices(
    labels: np.ndarray,
    *,
    region_ids: np.ndarray,
    dense_lookup: np.ndarray | None,
    dense_minimum: int,
) -> np.ndarray:
    """Map one label block onto compact row indices."""

    values = np.asarray(labels).ravel(order="C")
    if not np.issubdtype(values.dtype, np.integer):
        values = np.rint(values).astype(np.int64)
    if dense_lookup is not None:
        if dense_minimum == 0:
            return dense_lookup[values]
        return dense_lookup[values.astype(np.int64, copy=False) - dense_minimum]
    return np.searchsorted(region_ids, values)


def _aggregate_signal_by_region(
    signal_data: np.ndarray,
    annotation_data: np.ndarray,
    hemisphere_data: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate all regional voxel counts and fractional signal in two passes."""

    region_ids = _validate_and_collect_region_ids(annotation_data)
    if region_ids.size == 0:
        raise ValueError("Annotation contains no region labels")
    dense_lookup, dense_minimum = _build_region_indexer(region_ids)
    region_count = len(region_ids)
    scope_count = 3 if hemisphere_data is not None else 1
    grouped_counts = np.zeros(region_count * scope_count, dtype=np.int64)
    grouped_sums = np.zeros(region_count * scope_count, dtype=np.float64)

    for block_slice in _axis0_chunk_slices(
        annotation_data.shape,
        max_chunk_voxels=DEFAULT_MAX_AGGREGATION_CHUNK_VOXELS,
    ):
        compact_regions = _region_indices(
            annotation_data[block_slice],
            region_ids=region_ids,
            dense_lookup=dense_lookup,
            dense_minimum=dense_minimum,
        ).astype(np.int64, copy=False)
        signal_values = np.asarray(signal_data[block_slice]).ravel(order="C")
        if hemisphere_data is not None:
            hemisphere_values = np.asarray(hemisphere_data[block_slice]).ravel(order="C")
            scope_ids = np.zeros(hemisphere_values.shape, dtype=np.int64)
            scope_ids[hemisphere_values == LEFT_HEMISPHERE_ID] = 1
            scope_ids[hemisphere_values == RIGHT_HEMISPHERE_ID] = 2
            compact_regions *= scope_count
            compact_regions += scope_ids
        grouped_counts += np.bincount(
            compact_regions,
            minlength=region_count * scope_count,
        )
        grouped_sums += np.bincount(
            compact_regions,
            weights=signal_values,
            minlength=region_count * scope_count,
        )

    if hemisphere_data is None:
        return (
            region_ids,
            grouped_counts.reshape(region_count, 1),
            grouped_sums.reshape(region_count, 1),
        )

    counts_by_hemisphere = grouped_counts.reshape(region_count, scope_count)
    sums_by_hemisphere = grouped_sums.reshape(region_count, scope_count)
    # Scope zero holds voxels not marked left/right. These normally occur only
    # in annotation background, but bilateral totals must preserve them.
    bilateral_counts = counts_by_hemisphere.sum(axis=1)
    bilateral_sums = sums_by_hemisphere.sum(axis=1)
    region_counts = np.column_stack(
        (bilateral_counts, counts_by_hemisphere[:, 1], counts_by_hemisphere[:, 2])
    )
    signal_sums = np.column_stack(
        (bilateral_sums, sums_by_hemisphere[:, 1], sums_by_hemisphere[:, 2])
    )
    return region_ids, region_counts, signal_sums
