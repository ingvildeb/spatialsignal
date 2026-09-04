"""Quality-control helpers for point-cloud generation."""

from .centroid_images import pointcloud_slice_to_image, write_centroid_images
from .colocalization import write_colocalization_images, write_colocalization_report
from .deduplicate_pairs import (
    duplicate_status_colormap,
    read_mask_crop,
    render_duplicate_status_mask,
    render_duplicate_status_labels,
    select_qc_plane_pairs,
    write_pair_duplicate_qc,
)

__all__ = [
    "duplicate_status_colormap",
    "pointcloud_slice_to_image",
    "read_mask_crop",
    "render_duplicate_status_mask",
    "render_duplicate_status_labels",
    "select_qc_plane_pairs",
    "write_colocalization_images",
    "write_colocalization_report",
    "write_centroid_images",
    "write_pair_duplicate_qc",
]
