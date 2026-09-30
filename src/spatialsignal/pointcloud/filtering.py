"""Quality-control filters for point-cloud detections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from spatialsignal.models import (
    DataRepresentation,
    PointCloudDataset,
    ProcessingProvenance,
)


@dataclass(frozen=True)
class DetectionAreaFilterResult:
    """Accepted and rejected detections from a calibrated area filter."""

    accepted: PointCloudDataset
    rejected: PointCloudDataset
    maximum_area_um2: float
    pixel_area_um2: float
    source_detection_count: int

    @property
    def summary(self) -> dict[str, int | float | str]:
        """Return compact counts and the physical-area filtering contract."""

        return {
            "source_detections": self.source_detection_count,
            "accepted_detections": len(self.accepted.points),
            "rejected_detections": len(self.rejected.points),
            "maximum_area_um2": self.maximum_area_um2,
            "pixel_area_um2": self.pixel_area_um2,
            "accepted_rule": "area_um2 <= maximum_area_um2",
        }


def filter_detections_by_area(
    dataset: PointCloudDataset,
    *,
    maximum_area_um2: float,
    area_column: str = "area_px",
    source_name: str | None = None,
) -> DetectionAreaFilterResult:
    """Split raw detections using an inclusive calibrated-area upper limit.

    The source dataset and its metadata are never modified. Both returned
    datasets retain the original detection IDs and add an ``area_um2`` column.
    """

    dataset.validate()
    maximum_area_um2 = float(maximum_area_um2)
    if not np.isfinite(maximum_area_um2) or maximum_area_um2 <= 0:
        raise ValueError("maximum_area_um2 must be finite and positive")
    if area_column not in dataset.points.columns:
        raise ValueError(
            f"Point cloud is missing the detection-area column {area_column!r}"
        )

    resolution_by_axis = dict(
        zip(dataset.space.axis_labels, dataset.space.resolution_um, strict=True)
    )
    try:
        pixel_area_um2 = float(resolution_by_axis["x"]) * float(
            resolution_by_axis["y"]
        )
    except KeyError as exc:
        raise ValueError(
            "Point-cloud space must declare x and y axis resolutions"
        ) from exc
    if not np.isfinite(pixel_area_um2) or pixel_area_um2 <= 0:
        raise ValueError("Point-cloud x/y pixel area must be finite and positive")

    area_px = pd.to_numeric(dataset.points[area_column], errors="coerce")
    invalid_area = area_px.isna() | ~np.isfinite(area_px) | area_px.le(0)
    if invalid_area.any():
        invalid_ids = dataset.points.loc[invalid_area, "detection_id"].tolist()
        raise ValueError(
            f"{area_column} must contain finite positive values; invalid "
            f"detection IDs: {invalid_ids[:10]}"
        )

    area_um2 = area_px.astype(float) * pixel_area_um2
    accepted_mask = area_um2.le(maximum_area_um2)
    accepted_points = dataset.points.loc[accepted_mask].copy().reset_index(drop=True)
    rejected_points = dataset.points.loc[~accepted_mask].copy().reset_index(drop=True)
    accepted_points["area_um2"] = area_um2.loc[accepted_mask].to_numpy(dtype=float)
    rejected_points["area_um2"] = area_um2.loc[~accepted_mask].to_numpy(dtype=float)

    summary: dict[str, Any] = {
        "source_detections": int(len(dataset.points)),
        "accepted_detections": int(accepted_mask.sum()),
        "rejected_detections": int((~accepted_mask).sum()),
    }
    parameters: dict[str, Any] = {
        "area_column": area_column,
        "output_area_column": "area_um2",
        "maximum_area_um2": maximum_area_um2,
        "pixel_area_um2": pixel_area_um2,
        "accepted_rule": "area_um2 <= maximum_area_um2",
    }

    def derived_dataset(
        points: pd.DataFrame,
        *,
        population: str,
        representation_type: str,
    ) -> PointCloudDataset:
        metadata = replace(
            deepcopy(dataset.metadata),
            representation=DataRepresentation(
                kind="point_cloud",
                representation_type=representation_type,
            ),
            processing=ProcessingProvenance(
                stage="filter_detections_by_area",
                source_name=source_name,
                parameters={**parameters, "population": population},
                summary=summary.copy(),
            ),
        )
        return PointCloudDataset(
            subject_name=dataset.subject_name,
            points=points,
            metadata=metadata,
        )

    return DetectionAreaFilterResult(
        accepted=derived_dataset(
            accepted_points,
            population="accepted",
            representation_type="area_filtered_point_centroids",
        ),
        rejected=derived_dataset(
            rejected_points,
            population="rejected",
            representation_type="area_rejected_point_centroids",
        ),
        maximum_area_um2=maximum_area_um2,
        pixel_area_um2=pixel_area_um2,
        source_detection_count=len(dataset.points),
    )
