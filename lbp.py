from skimage.feature import local_binary_pattern
import numpy as np
import matplotlib.pyplot as plt

def compute_lbp_for_single_image(image, radius, n_points, method='nri_uniform'):
    return local_binary_pattern(image, n_points, radius, method)

def compute_lbp_for_many_images(images, radius, n_points, method='nri_uniform'):
    return [compute_lbp_for_single_image(img, radius, n_points, method) for img in images]

def compute_lbp_for_each_category(categories, categories_images, radius=2, n_points=8, method='nri_uniform'):
    return {category: compute_lbp_for_many_images(categories_images[category], radius, n_points, method) for category in categories}

def build_histogram_from_lbp(lbp, n_pixels):
    n_bins = int(lbp.max() + 1)
    histogram, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    # normalize the histogram
    histogram = histogram.astype("float")
    histogram /= histogram.sum() + 1e-6

    return histogram


def build_histogram_from_many_lbps(lbps, n_pixels):
    return [build_histogram_from_lbp(lbp, n_pixels) for lbp in lbps]


def build_histograms_from_categories(categories, lbps, n_pixels):
    return {
        category: build_histogram_from_many_lbps(lbps[category], n_pixels)
        for category in categories
    }

