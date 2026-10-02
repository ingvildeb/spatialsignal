"""Quality-control filters for point-cloud detections."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import operator
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from spatialsignal.models import (
    DataRepresentation,
    PointCloudDataset,
    ProcessingProvenance,
)


ConditionCombination = Literal["any", "all"]
HARD_AREA_REJECTION_REASON = "area_above_hard_threshold"
CONDITIONAL_REJECTION_REASON = "conditional_area_and_conditions_met"

_CONDITION_OPERATORS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


@dataclass(frozen=True)
class DetectionFilterCondition:
    """One explicit numeric feature comparison in a conditional area rule."""

    feature: str
    operator: str
    threshold: float

    def __post_init__(self) -> None:
        feature = str(self.feature).strip()
        comparison = str(self.operator).strip()
        threshold = float(self.threshold)
        if not feature:
            raise ValueError("condition feature must be non-empty")
        if comparison not in _CONDITION_OPERATORS:
            supported = ", ".join(sorted(_CONDITION_OPERATORS))
            raise ValueError(
                f"condition operator must be one of {supported}; "
                f"received {comparison!r}"
            )
        if not np.isfinite(threshold):
            raise ValueError("condition threshold must be finite")
        object.__setattr__(self, "feature", feature)
        object.__setattr__(self, "operator", comparison)
        object.__setattr__(self, "threshold", threshold)

    @property
    def expression(self) -> str:
        """Return the human-readable comparison recorded in provenance."""

        return f"{self.feature} {self.operator} {self.threshold:g}"

    def to_dict(self) -> dict[str, str | float]:
        """Return a JSON-serializable condition record."""

        return {
            "feature": self.feature,
            "operator": self.operator,
            "threshold": self.threshold,
        }


@dataclass(frozen=True)
class DetectionMorphologyFilterResult:
    """Accepted and rejected detections from one morphology-QC operation."""

    accepted: PointCloudDataset
    rejected: PointCloudDataset
    hard_area_threshold_um2: float
    conditional_area_threshold_um2: float | None
    conditions: tuple[DetectionFilterCondition, ...]
    condition_combination: ConditionCombination
    pixel_area_um2: float
    source_detection_count: int

    @property
    def maximum_area_um2(self) -> float:
        """Return the hard threshold under the legacy area-filter name."""

        return self.hard_area_threshold_um2

    @property
    def summary(self) -> dict[str, Any]:
        """Return counts and the complete physical-morphology filter contract."""

        rejected_reasons = self.rejected.points["rejection_reason"]
        return {
            "source_detections": self.source_detection_count,
            "accepted_detections": len(self.accepted.points),
            "rejected_detections": len(self.rejected.points),
            "rejected_hard_area": int(
                rejected_reasons.eq(HARD_AREA_REJECTION_REASON).sum()
            ),
            "rejected_conditional": int(
                rejected_reasons.eq(CONDITIONAL_REJECTION_REASON).sum()
            ),
            "hard_area_threshold_um2": self.hard_area_threshold_um2,
            "conditional_area_threshold_um2": (
                self.conditional_area_threshold_um2
            ),
            "conditions": [condition.to_dict() for condition in self.conditions],
            "condition_combination": self.condition_combination,
            "pixel_area_um2": self.pixel_area_um2,
            "accepted_rule": _accepted_rule(
                self.hard_area_threshold_um2,
                self.conditional_area_threshold_um2,
                self.conditions,
                self.condition_combination,
            ),
        }


# The legacy result name remains a type alias because the area-only convenience
# function delegates to the same composite implementation.
DetectionAreaFilterResult = DetectionMorphologyFilterResult


def _accepted_rule(
    hard_area_threshold_um2: float,
    conditional_area_threshold_um2: float | None,
    conditions: tuple[DetectionFilterCondition, ...],
    condition_combination: ConditionCombination,
) -> str:
    hard_rule = f"area_um2 <= {hard_area_threshold_um2:g}"
    if conditional_area_threshold_um2 is None:
        return hard_rule
    connector = "or" if condition_combination == "any" else "and"
    condition_expression = f" {connector} ".join(
        f"({condition.expression})" for condition in conditions
    )
    return (
        f"{hard_rule} and not (area_um2 > "
        f"{conditional_area_threshold_um2:g} and "
        f"{condition_expression})"
    )


def _pixel_area_um2(dataset: PointCloudDataset) -> float:
    resolution_by_axis = dict(
        zip(dataset.space.axis_labels, dataset.space.resolution_um, strict=True)
    )
    try:
        pixel_area = float(resolution_by_axis["x"]) * float(
            resolution_by_axis["y"]
        )
    except KeyError as exc:
        raise ValueError(
            "Point-cloud space must declare x and y axis resolutions"
        ) from exc
    if not np.isfinite(pixel_area) or pixel_area <= 0:
        raise ValueError("Point-cloud x/y pixel area must be finite and positive")
    return pixel_area


def _numeric_feature(
    dataset: PointCloudDataset,
    condition: DetectionFilterCondition,
) -> pd.Series:
    if condition.feature not in dataset.points.columns:
        raise ValueError(
            f"Point cloud is missing condition feature {condition.feature!r}"
        )
    values = pd.to_numeric(dataset.points[condition.feature], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any():
        invalid_ids = dataset.points.loc[invalid, "detection_id"].tolist()
        raise ValueError(
            f"condition feature {condition.feature!r} must contain finite "
            f"numeric values; invalid detection IDs: {invalid_ids[:10]}"
        )
    values = values.astype(float)
    if condition.feature == "eccentricity" and (
        values.lt(0).any() or values.gt(1).any()
    ):
        invalid_ids = dataset.points.loc[
            values.lt(0) | values.gt(1), "detection_id"
        ].tolist()
        raise ValueError(
            "eccentricity must contain values from 0 to 1; invalid detection "
            f"IDs: {invalid_ids[:10]}"
        )
    return values


def filter_detections_by_morphology(
    dataset: PointCloudDataset,
    *,
    hard_area_threshold_um2: float,
    conditional_area_threshold_um2: float | None = None,
    conditions: Sequence[DetectionFilterCondition] | None = None,
    condition_combination: ConditionCombination = "any",
    area_column: str = "area_px",
    source_name: str | None = None,
) -> DetectionMorphologyFilterResult:
    """Split detections using one calibrated-area and feature-QC operation.

    A detection is rejected when its physical area is above the hard threshold,
    or when it is above the conditional threshold and the configured feature
    expression is true. The source dataset and metadata are never modified.
    Both returned populations retain source detection IDs and add ``area_um2``;
    rejected detections also receive a mutually exclusive ``rejection_reason``.
    """

    dataset.validate()
    hard_area_threshold_um2 = float(hard_area_threshold_um2)
    if (
        not np.isfinite(hard_area_threshold_um2)
        or hard_area_threshold_um2 <= 0
    ):
        raise ValueError("hard_area_threshold_um2 must be finite and positive")
    if condition_combination not in {"any", "all"}:
        raise ValueError("condition_combination must be 'any' or 'all'")

    normalized_conditions = tuple(conditions or ())
    if any(
        not isinstance(condition, DetectionFilterCondition)
        for condition in normalized_conditions
    ):
        raise TypeError(
            "conditions must contain only DetectionFilterCondition instances"
        )
    has_conditional_area = conditional_area_threshold_um2 is not None
    has_conditions = bool(normalized_conditions)
    if has_conditional_area != has_conditions:
        raise ValueError(
            "conditional_area_threshold_um2 and conditions must be provided "
            "together"
        )
    if has_conditional_area:
        conditional_area_threshold_um2 = float(
            conditional_area_threshold_um2
        )
        if (
            not np.isfinite(conditional_area_threshold_um2)
            or conditional_area_threshold_um2 <= 0
            or conditional_area_threshold_um2 >= hard_area_threshold_um2
        ):
            raise ValueError(
                "conditional_area_threshold_um2 must be finite, positive, and "
                "below hard_area_threshold_um2"
            )

    if area_column not in dataset.points.columns:
        raise ValueError(
            f"Point cloud is missing the detection-area column {area_column!r}"
        )
    area_px = pd.to_numeric(dataset.points[area_column], errors="coerce")
    invalid_area = area_px.isna() | ~np.isfinite(area_px) | area_px.le(0)
    if invalid_area.any():
        invalid_ids = dataset.points.loc[invalid_area, "detection_id"].tolist()
        raise ValueError(
            f"{area_column} must contain finite positive values; invalid "
            f"detection IDs: {invalid_ids[:10]}"
        )

    pixel_area = _pixel_area_um2(dataset)
    area_um2 = area_px.astype(float) * pixel_area
    hard_rejected = area_um2.gt(hard_area_threshold_um2)
    conditional_rejected = pd.Series(False, index=dataset.points.index)
    if normalized_conditions:
        condition_masks = [
            pd.Series(
                _CONDITION_OPERATORS[condition.operator](
                    _numeric_feature(dataset, condition),
                    condition.threshold,
                ),
                index=dataset.points.index,
            )
            for condition in normalized_conditions
        ]
        combined_conditions = pd.concat(condition_masks, axis=1)
        condition_met = (
            combined_conditions.any(axis=1)
            if condition_combination == "any"
            else combined_conditions.all(axis=1)
        )
        conditional_rejected = (
            ~hard_rejected
            & area_um2.gt(float(conditional_area_threshold_um2))
            & condition_met
        )

    rejected_mask = hard_rejected | conditional_rejected
    accepted_mask = ~rejected_mask
    accepted_points = dataset.points.loc[accepted_mask].copy().reset_index(drop=True)
    rejected_points = dataset.points.loc[rejected_mask].copy().reset_index(drop=True)
    accepted_points["area_um2"] = area_um2.loc[accepted_mask].to_numpy(dtype=float)
    rejected_points["area_um2"] = area_um2.loc[rejected_mask].to_numpy(dtype=float)
    rejection_reasons = np.full(len(dataset.points), "", dtype=object)
    rejection_reasons[hard_rejected.to_numpy()] = HARD_AREA_REJECTION_REASON
    rejection_reasons[conditional_rejected.to_numpy()] = (
        CONDITIONAL_REJECTION_REASON
    )
    rejected_points["rejection_reason"] = rejection_reasons[
        rejected_mask.to_numpy()
    ]

    accepted_rule = _accepted_rule(
        hard_area_threshold_um2,
        conditional_area_threshold_um2,
        normalized_conditions,
        condition_combination,
    )
    summary: dict[str, Any] = {
        "source_detections": int(len(dataset.points)),
        "accepted_detections": int(accepted_mask.sum()),
        "rejected_detections": int(rejected_mask.sum()),
        "rejected_hard_area": int(hard_rejected.sum()),
        "rejected_conditional": int(conditional_rejected.sum()),
    }
    parameters: dict[str, Any] = {
        "area_column": area_column,
        "output_area_column": "area_um2",
        "hard_area_threshold_um2": hard_area_threshold_um2,
        "conditional_area_threshold_um2": conditional_area_threshold_um2,
        "conditions": [
            condition.to_dict() for condition in normalized_conditions
        ],
        "condition_combination": condition_combination,
        "pixel_area_um2": pixel_area,
        "accepted_rule": accepted_rule,
        "rejection_reasons": {
            "hard_area": HARD_AREA_REJECTION_REASON,
            "conditional": CONDITIONAL_REJECTION_REASON,
        },
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
                stage="filter_detections_by_morphology",
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

    return DetectionMorphologyFilterResult(
        accepted=derived_dataset(
            accepted_points,
            population="accepted",
            representation_type="morphology_filtered_point_centroids",
        ),
        rejected=derived_dataset(
            rejected_points,
            population="rejected",
            representation_type="morphology_rejected_point_centroids",
        ),
        hard_area_threshold_um2=hard_area_threshold_um2,
        conditional_area_threshold_um2=conditional_area_threshold_um2,
        conditions=normalized_conditions,
        condition_combination=condition_combination,
        pixel_area_um2=pixel_area,
        source_detection_count=len(dataset.points),
    )


def filter_detections_by_area(
    dataset: PointCloudDataset,
    *,
    maximum_area_um2: float,
    area_column: str = "area_px",
    source_name: str | None = None,
) -> DetectionAreaFilterResult:
    """Apply the composite engine with only a hard calibrated-area threshold."""

    return filter_detections_by_morphology(
        dataset,
        hard_area_threshold_um2=maximum_area_um2,
        area_column=area_column,
        source_name=source_name,
    )
