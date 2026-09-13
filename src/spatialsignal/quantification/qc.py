"""Optional visual QC for subject-space atlas-region quantification."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from spatialsignal.integration import LabelVolume
from spatialsignal.models import PointCloudDataset, SpaceDefinition

from .instance_regions import _validate_same_sampling_space
from .region_quantification import RegionQuantificationResult

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def write_region_quantification_qc(
    quantification: RegionQuantificationResult | PointCloudDataset,
    annotation: LabelVolume,
    output_path: Path,
    *,
    title: str | None = None,
    background_id: int = 0,
    out_of_bounds_id: int = -1,
    max_points_per_slice: int = 20_000,
    dpi: int = 150,
) -> Path:
    """Write projection and slice overlays for region-assignment QC.

    ``quantification`` may be a newly returned ``RegionQuantificationResult``
    or a saved region-assigned ``PointCloudDataset`` loaded from its Parquet and
    metadata JSON files. Coordinates must already be expressed in the
    annotation sampling grid.
    """

    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Visual quantification QC requires the spatialsignal matplotlib dependency."
        ) from exc

    if max_points_per_slice < 1:
        raise ValueError("max_points_per_slice must be at least 1")
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"QC output_path must have a .png suffix, got {output_path}")

    assigned, object_space, subject_name = _resolve_qc_input(quantification)
    _validate_qc_table(assigned)
    _validate_same_sampling_space(
        object_space,
        annotation.space,
        volume_name="annotation",
    )
    if tuple(annotation.data.shape) != tuple(annotation.space.shape):
        raise ValueError(
            "Annotation array shape does not match its declared space: "
            f"{annotation.data.shape} vs {tuple(annotation.space.shape)}"
        )

    indices, float_coords = _coordinates_in_zero_based_annotation_grid(
        assigned,
        object_space,
    )
    shape = tuple(int(value) for value in annotation.space.shape)
    in_bounds = _in_bounds(indices, shape)
    region_ids = assigned["region_id"].to_numpy(dtype=np.int64, copy=False)
    assigned_mask = (
        in_bounds
        & (region_ids != background_id)
        & (region_ids != out_of_bounds_id)
    )
    background_mask = in_bounds & (region_ids == background_id)
    out_of_bounds_mask = (~in_bounds) | (region_ids == out_of_bounds_id)

    plane_specs = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 2, 0),
    )
    foreground = np.asarray(annotation.data) != background_id
    slice_indices = _representative_slice_indices(indices, in_bounds, shape)

    figure = Figure(figsize=(18, 11), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    orientation = annotation.space.orientation.lower()
    for column, (horizontal_axis, vertical_axis, collapsed_axis) in enumerate(
        plane_specs
    ):
        projection_ax = axes[0, column]
        density = _count_projection(
            indices,
            in_bounds,
            horizontal_axis,
            vertical_axis,
            shape,
        )
        image = projection_ax.imshow(
            np.log1p(density).T,
            origin="lower",
            interpolation="nearest",
            cmap="magma",
            aspect="equal",
        )
        foreground_projection = np.any(foreground, axis=collapsed_axis)
        _draw_binary_contour(projection_ax, foreground_projection, color="cyan")
        projection_ax.set_title(
            "Point-count projection\n"
            f"{_axis_direction(orientation[horizontal_axis])} × "
            f"{_axis_direction(orientation[vertical_axis])}; "
            f"collapse {_axis_direction(orientation[collapsed_axis])}"
        )
        _label_axes(projection_ax, orientation, horizontal_axis, vertical_axis)
        figure.colorbar(image, ax=projection_ax, shrink=0.75, label="log(1 + objects)")

        slice_ax = axes[1, column]
        slice_index = slice_indices[collapsed_axis]
        annotation_slice = np.take(annotation.data, slice_index, axis=collapsed_axis)
        slice_ax.imshow(
            (annotation_slice != background_id).T,
            origin="lower",
            interpolation="nearest",
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.55,
            aspect="equal",
        )
        boundaries = find_boundaries(annotation_slice, connectivity=1, mode="thick")
        if np.any(boundaries) and min(boundaries.shape) >= 2:
            slice_ax.contour(
                boundaries.T.astype(float),
                levels=[0.5],
                colors="black",
                linewidths=0.3,
                alpha=0.65,
            )
        slice_mask = in_bounds & (indices[collapsed_axis] == slice_index)
        _scatter_slice_points(
            slice_ax,
            float_coords,
            slice_mask & assigned_mask,
            horizontal_axis,
            vertical_axis,
            max_points=max_points_per_slice,
            color="#00d8ff",
            label="assigned",
        )
        _scatter_slice_points(
            slice_ax,
            float_coords,
            slice_mask & background_mask,
            horizontal_axis,
            vertical_axis,
            max_points=max_points_per_slice,
            color="red",
            label="background",
        )
        slice_ax.set_title(
            "Annotation slice\n"
            f"{_axis_direction(orientation[collapsed_axis])} index {slice_index}"
        )
        _label_axes(slice_ax, orientation, horizontal_axis, vertical_axis)
        handles, labels = slice_ax.get_legend_handles_labels()
        if handles:
            slice_ax.legend(loc="upper right", fontsize=8, markerscale=3)

    n_objects = int(len(assigned))
    n_assigned = int(np.count_nonzero(assigned_mask))
    n_background = int(np.count_nonzero(background_mask))
    n_out_of_bounds = int(np.count_nonzero(out_of_bounds_mask))
    display_title = title or f"{subject_name}: atlas-region quantification QC"
    summary = (
        f"objects={n_objects:,} | assigned={n_assigned:,} "
        f"({_percentage(n_assigned, n_objects):.3f}%) | "
        f"background={n_background:,} "
        f"({_percentage(n_background, n_objects):.3f}%) | "
        f"out of bounds={n_out_of_bounds:,} "
        f"({_percentage(n_out_of_bounds, n_objects):.3f}%)\n"
        f"annotation shape={shape}, orientation={annotation.space.orientation}, "
        "resolution="
        + " × ".join(f"{float(value):.4g}" for value in annotation.space.resolution_um)
        + " µm"
    )
    figure.suptitle(f"{display_title}\n{summary}", fontsize=13)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor="white")
    return output_path


def _resolve_qc_input(
    quantification: RegionQuantificationResult | PointCloudDataset,
) -> tuple[pd.DataFrame, SpaceDefinition, str]:
    if isinstance(quantification, RegionQuantificationResult):
        return (
            quantification.assigned_objects,
            quantification.remapped_objects.space,
            quantification.remapped_objects.subject_name,
        )
    if isinstance(quantification, PointCloudDataset):
        return quantification.points, quantification.space, quantification.subject_name
    raise TypeError(
        "quantification must be a RegionQuantificationResult or PointCloudDataset"
    )


def _validate_qc_table(assigned: pd.DataFrame) -> None:
    missing = sorted({"x", "y", "z", "region_id"}.difference(assigned.columns))
    if missing:
        raise ValueError(f"Assigned-object table is missing QC columns: {missing}")


def _coordinates_in_zero_based_annotation_grid(
    assigned: pd.DataFrame,
    space: SpaceDefinition,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, ...]]:
    float_columns = ("x_float", "y_float", "z_float")
    if set(float_columns) <= set(assigned.columns):
        offset = 1.0 if space.indexing == "one_based" else 0.0
        float_coords = tuple(
            assigned[column].to_numpy(dtype=np.float64, copy=False) - offset
            for column in float_columns
        )
        indices = tuple(
            np.floor(coordinates + 0.5).astype(np.int64)
            for coordinates in float_coords
        )
        return indices, float_coords

    integer_offset = 1 if space.indexing == "one_based" else 0
    raw_indices = tuple(
        assigned[column].to_numpy(dtype=np.int64, copy=False)
        for column in ("x", "y", "z")
    )
    indices = (
        tuple(coordinates - integer_offset for coordinates in raw_indices)
        if integer_offset
        else raw_indices
    )
    float_coords = indices
    return indices, float_coords


def _in_bounds(
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int, int],
) -> np.ndarray:
    return (
        (indices[0] >= 0)
        & (indices[0] < shape[0])
        & (indices[1] >= 0)
        & (indices[1] < shape[1])
        & (indices[2] >= 0)
        & (indices[2] < shape[2])
    )


def _count_projection(
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    horizontal_axis: int,
    vertical_axis: int,
    shape: tuple[int, int, int],
) -> np.ndarray:
    linear_indices = (
        indices[horizontal_axis][mask] * shape[vertical_axis]
        + indices[vertical_axis][mask]
    )
    return np.bincount(
        linear_indices,
        minlength=shape[horizontal_axis] * shape[vertical_axis],
    ).reshape(shape[horizontal_axis], shape[vertical_axis])


def _representative_slice_indices(
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    in_bounds: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if not np.any(in_bounds):
        return tuple(int(length // 2) for length in shape)
    return tuple(int(np.median(axis_indices[in_bounds])) for axis_indices in indices)


def _draw_binary_contour(ax: "Axes", mask: np.ndarray, *, color: str) -> None:
    if min(mask.shape) >= 2 and np.any(mask) and not np.all(mask):
        ax.contour(
            mask.T.astype(float),
            levels=[0.5],
            colors=color,
            linewidths=0.8,
        )


def _scatter_slice_points(
    ax: "Axes",
    float_coords: tuple[np.ndarray, ...],
    mask: np.ndarray,
    horizontal_axis: int,
    vertical_axis: int,
    *,
    max_points: int,
    color: str,
    label: str,
) -> None:
    positions = np.flatnonzero(mask)
    if len(positions) > max_points:
        positions = positions[
            np.linspace(0, len(positions) - 1, max_points, dtype=np.int64)
        ]
    if len(positions):
        ax.scatter(
            float_coords[horizontal_axis][positions],
            float_coords[vertical_axis][positions],
            s=1.0,
            c=color,
            alpha=0.45,
            linewidths=0,
            label=label,
        )


def _label_axes(
    ax: "Axes",
    orientation: str,
    horizontal_axis: int,
    vertical_axis: int,
) -> None:
    ax.set_xlabel(
        f"axis {horizontal_axis} ({_axis_direction(orientation[horizontal_axis])})"
    )
    ax.set_ylabel(
        f"axis {vertical_axis} ({_axis_direction(orientation[vertical_axis])})"
    )


def _axis_direction(origin_letter: str) -> str:
    opposite = {
        "l": "r",
        "r": "l",
        "a": "p",
        "p": "a",
        "s": "i",
        "i": "s",
    }
    normalized = origin_letter.lower()
    return f"{normalized.upper()}→{opposite[normalized].upper()}"


def _percentage(count: int, total: int) -> float:
    return (100.0 * float(count) / float(total)) if total else 0.0
