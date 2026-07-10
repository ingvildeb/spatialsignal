"""Point-cloud construction utilities."""

from .build import build_pointcloud_from_masks, build_signal_points_from_masks
from .centroids import (
    POINTCLOUD_COLUMNS,
    POINTCLOUD_OPTIONAL_COLUMNS,
    POINTCLOUD_REQUIRED_COLUMNS,
    extract_centroids_from_mask,
    extract_centroids_from_mask_stack,
    matlab_round,
)
from .signal_points import (
    SIGNAL_POINT_REQUIRED_COLUMNS,
    extract_signal_points_from_mask,
    extract_signal_points_from_mask_stack,
)
from lsfm_cell_mapping.models.datasets import PointCloudDataset
from .deduplicate import (
    CLEANED_OBJECT_REQUIRED_COLUMNS,
    EDGE_COLUMNS,
    DeduplicationResult,
    aggregate_cleaned_objects,
    build_cross_plane_edge_table,
    build_object_membership_table,
    deduplicate_across_planes,
    summarize_deduplication_result,
)
from lsfm_cell_mapping.models.metadata import (
    DataRepresentation,
    DatasetMetadata,
    ProcessingProvenance,
    SpaceDefinition,
    build_dataset_metadata,
    write_dataset_metadata,
)
from .validate import (
    ValidationReport,
    compare_pointcloud_tables,
    load_canonical_pointcloud_csv,
    load_legacy_matlab_centroids_csv,
    relabel_slices_in_natural_order,
)

__all__ = [
    "build_pointcloud_from_masks",
    "build_signal_points_from_masks",
    "POINTCLOUD_COLUMNS",
    "POINTCLOUD_OPTIONAL_COLUMNS",
    "POINTCLOUD_REQUIRED_COLUMNS",
    "SIGNAL_POINT_REQUIRED_COLUMNS",
    "PointCloudDataset",
    "DataRepresentation",
    "DatasetMetadata",
    "ProcessingProvenance",
    "SpaceDefinition",
    "CLEANED_OBJECT_REQUIRED_COLUMNS",
    "DeduplicationResult",
    "EDGE_COLUMNS",
    "ValidationReport",
    "aggregate_cleaned_objects",
    "build_dataset_metadata",
    "build_cross_plane_edge_table",
    "build_object_membership_table",
    "compare_pointcloud_tables",
    "deduplicate_across_planes",
    "extract_centroids_from_mask",
    "extract_centroids_from_mask_stack",
    "extract_signal_points_from_mask",
    "extract_signal_points_from_mask_stack",
    "load_canonical_pointcloud_csv",
    "load_legacy_matlab_centroids_csv",
    "matlab_round",
    "relabel_slices_in_natural_order",
    "summarize_deduplication_result",
    "write_dataset_metadata",
]
