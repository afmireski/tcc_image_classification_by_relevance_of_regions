"""
Tools package for image classification project.

This package contains utility modules for image processing and analysis.
"""

from .image_tools import (
    segment_image_into_grid,
    segment_image_dynamic,
    segment_image_auto,
    calculate_optimal_grid,
    calculate_dynamic_k,
    segment_images_batch,
    segment_images_by_category,
    segment_images_batch_dynamic,
    segment_images_by_category_dynamic,
    segment_images_batch_auto,
    segment_images_by_category_auto,
    visualize_image_segmentation,
    visualize_image_segmentation_dynamic,
    visualize_image_segmentation_auto,
    get_region_statistics,
    reconstruct_image_from_regions
)

from .specialists import (
    build_specialist_set,
    build_specialist_set_for_many_classes,
    train_specialists
)

__all__ = [
    "segment_image_into_grid",
    "segment_image_dynamic", 
    "segment_image_auto",
    "calculate_optimal_grid",
    "calculate_dynamic_k",
    "segment_images_batch",
    "segment_images_by_category",
    "segment_images_batch_dynamic",
    "segment_images_by_category_dynamic",
    "segment_images_batch_auto",
    "segment_images_by_category_auto",
    "visualize_image_segmentation",
    "visualize_image_segmentation_dynamic",
    "visualize_image_segmentation_auto",
    "get_region_statistics",
    "reconstruct_image_from_regions",

    "build_specialist_set",
    "build_specialist_set_for_many_classes",
    "train_specialists"
]
