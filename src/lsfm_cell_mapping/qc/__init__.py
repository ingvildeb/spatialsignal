"""Quality-control helpers for point-cloud generation."""

from .centroid_images import pointcloud_slice_to_image, write_centroid_images

__all__ = ["pointcloud_slice_to_image", "write_centroid_images"]
