"""Quality-control helpers for point-cloud generation."""

from .centroid_images import pointcloud_slice_to_image, write_centroid_images
from .deduplicate_pairs import select_qc_plane_pairs, write_pair_duplicate_qc

__all__ = [
    "pointcloud_slice_to_image",
    "select_qc_plane_pairs",
    "write_centroid_images",
    "write_pair_duplicate_qc",
]
