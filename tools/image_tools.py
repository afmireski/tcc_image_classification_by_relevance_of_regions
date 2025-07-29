"""
Image Tools Module

This module provides utilities for image processing, specifically for segmenting images
into regions based on a segmentation factor.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any


def segment_image_into_grid(
    image: np.ndarray, segmentation_factor: int
) -> np.ndarray:
    """
    Segments an image into a grid of regions based on the segmentation factor.

    Args:
        image (np.ndarray): Input grayscale image as numpy array
        segmentation_factor (int): Number of segments per dimension (creates segmentation_factor x segmentation_factor grid)

    Returns:
        np.ndarray: A segmentation_factor x segmentation_factor matrix where each element
                   contains the pixel values of that region as a numpy array
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be grayscale (2D array)")

    height, width = image.shape

    # Calculate the size of each region
    region_height = height // segmentation_factor
    region_width = width // segmentation_factor

    # Create a matrix to hold the regions
    regions_matrix = np.empty((segmentation_factor, segmentation_factor), dtype=object)

    for row_idx in range(segmentation_factor):
        for col_idx in range(segmentation_factor):
            # Calculate coordinates for this region
            start_row = row_idx * region_height
            end_row = start_row + region_height
            start_col = col_idx * region_width
            end_col = start_col + region_width

            # Handle edge cases for the last regions if image dimensions aren't perfectly divisible
            if row_idx == segmentation_factor - 1:
                end_row = height
            if col_idx == segmentation_factor - 1:
                end_col = width

            # Extract the region and store in matrix
            region = image[start_row:end_row, start_col:end_col]
            regions_matrix[row_idx, col_idx] = region

    return regions_matrix


def segment_images_batch(
    images: List[np.ndarray], segmentation_factor: int
) -> List[np.ndarray]:
    """
    Segments a batch of images into regions.

    Args:
        images (List[np.ndarray]): List of grayscale images
        segmentation_factor (int): Number of segments per dimension

    Returns:
        List[np.ndarray]: List of segmentation matrices, one for each input image.
                         Each matrix is segmentation_factor x segmentation_factor with
                         pixel arrays in each cell.
    """
    all_segmented_images = []

    for i, image in enumerate(images):
        try:
            segmented_regions = segment_image_into_grid(image, segmentation_factor)
            all_segmented_images.append(segmented_regions)
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            # Create empty matrix for failed images
            empty_matrix = np.empty((segmentation_factor, segmentation_factor), dtype=object)
            all_segmented_images.append(empty_matrix)

    return all_segmented_images


def segment_images_by_category(
    images_by_category: Dict[str, List[np.ndarray]], segmentation_factor: int
) -> Dict[str, List[np.ndarray]]:
    """
    Segments images organized by category.

    Args:
        images_by_category (Dict[str, List[np.ndarray]]): Dictionary with category names as keys
                                                         and lists of images as values
        segmentation_factor (int): Number of segments per dimension

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary with category names as keys and
                                   lists of segmentation matrices as values
    """
    segmented_by_category = {}

    for category, images in images_by_category.items():
        print(f"Segmenting {len(images)} images from category '{category}'...")
        segmented_images = segment_images_batch(images, segmentation_factor)
        segmented_by_category[category] = segmented_images

        # Print summary
        total_regions = len(segmented_images) * segmentation_factor * segmentation_factor
        print(
            f"  Generated {total_regions} regions ({segmentation_factor}x{segmentation_factor} grid per image)"
        )

    return segmented_by_category


def visualize_image_segmentation(
    image: np.ndarray, segmentation_factor: int, figsize: Tuple[int, int] = (12, 8)
):
    """
    Visualizes the original image and its segmented regions.

    Args:
        image (np.ndarray): Input grayscale image
        segmentation_factor (int): Number of segments per dimension
        figsize (Tuple[int, int]): Figure size for matplotlib
    """
    regions_matrix = segment_image_into_grid(image, segmentation_factor)

    # Calculate subplot grid  
    cols = segmentation_factor
    rows = segmentation_factor

    plt.figure(figsize=figsize)

    # Show each region
    plot_idx = 1
    for row_idx in range(segmentation_factor):
        for col_idx in range(segmentation_factor):
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(regions_matrix[row_idx, col_idx], cmap="gray")
            plt.title(f"Region ({row_idx},{col_idx})")
            plt.axis("off")
            plot_idx += 1

    plt.tight_layout()
    plt.show()


def get_region_statistics(regions_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Calculate statistics for a matrix of image regions.

    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions (segmentation_factor x segmentation_factor)

    Returns:
        Dict[str, Any]: Statistics including min/max/mean dimensions and total regions
    """
    # Check if the matrix is empty or has no dimensions
    if regions_matrix.shape == (0,) or len(regions_matrix.shape) == 0:
        return {}

    # Get all region shapes
    region_shapes = []
    total_regions = 0
    
    for row_idx in range(regions_matrix.shape[0]):
        for col_idx in range(regions_matrix.shape[1]):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, 'shape'):
                region_shapes.append(region.shape)
                total_regions += 1

    if not region_shapes:
        return {}

    heights = [shape[0] for shape in region_shapes]
    widths = [shape[1] for shape in region_shapes]

    stats = {
        "total_regions": total_regions,
        "grid_shape": regions_matrix.shape,
        "min_height": min(heights),
        "max_height": max(heights),
        "mean_height": np.mean(heights),
        "min_width": min(widths),
        "max_width": max(widths),
        "mean_width": np.mean(widths),
    }

    return stats


def reconstruct_image_from_regions(regions_matrix: np.ndarray) -> np.ndarray:
    """
    Reconstructs an image from its segmented regions matrix.

    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions (segmentation_factor x segmentation_factor)

    Returns:
        np.ndarray: Reconstructed image
    """
    # Check if the matrix is empty or has no dimensions
    if regions_matrix.shape == (0,) or len(regions_matrix.shape) == 0:
        raise ValueError("No regions provided for reconstruction")

    segmentation_factor = regions_matrix.shape[0]
    
    # Get the first non-None region to determine data type
    first_region = None
    for row_idx in range(segmentation_factor):
        for col_idx in range(segmentation_factor):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, 'shape'):
                first_region = region
                break
        if first_region is not None:
            break
    
    if first_region is None:
        raise ValueError("All regions are None")

    # Calculate total image dimensions
    region_height, region_width = first_region.shape
    total_height = region_height * segmentation_factor
    total_width = region_width * segmentation_factor
    
    # Create the reconstructed image
    reconstructed = np.zeros((total_height, total_width), dtype=first_region.dtype)
    
    # Fill in each region
    for row_idx in range(segmentation_factor):
        for col_idx in range(segmentation_factor):
            region = regions_matrix[row_idx, col_idx]
            if region is not None and hasattr(region, 'shape'):
                start_row = row_idx * region_height
                end_row = start_row + region.shape[0]
                start_col = col_idx * region_width
                end_col = start_col + region.shape[1]
                
                reconstructed[start_row:end_row, start_col:end_col] = region

    return reconstructed


def get_region(regions_matrix: np.ndarray, row: int, col: int) -> np.ndarray:
    """
    Get a specific region from the segmentation matrix.
    
    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions
        row (int): Row index of the region
        col (int): Column index of the region
    
    Returns:
        np.ndarray: The region's pixel data
    """
    return regions_matrix[row, col]


def show_region_example(regions_matrix: np.ndarray, row: int = 0, col: int = 0):
    """
    Display a specific region from the segmentation matrix.
    
    Args:
        regions_matrix (np.ndarray): Matrix of segmented regions
        row (int): Row index of the region to show
        col (int): Column index of the region to show
    """
    region = get_region(regions_matrix, row, col)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(region, cmap='gray')
    plt.title(f'Region ({row}, {col}) - Shape: {region.shape}')
    plt.axis('off')
    plt.show()
