"""Atlas-aware subject-space quantification helpers."""

from .atlas_regions import enrich_region_summary_with_atlas
from .instance_regions import (
    assign_objects_to_regions,
    remap_objects_to_space,
    remap_pointcloud_dataset_to_space,
    summarize_objects_by_region,
    validate_subject_space_match,
)

__all__ = [
    "assign_objects_to_regions",
    "enrich_region_summary_with_atlas",
    "remap_objects_to_space",
    "remap_pointcloud_dataset_to_space",
    "summarize_objects_by_region",
    "validate_subject_space_match",
]
