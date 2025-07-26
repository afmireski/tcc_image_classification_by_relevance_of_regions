from skimage.feature import graycomatrix, graycoprops
import numpy as np
from joblib import Parallel, delayed

def calculate_glcm_from_single_image(
    image, distances, angles, levels, symmetric=True, normed=True
):
    return graycomatrix(
        image, distances, angles, levels, symmetric=symmetric, normed=normed
    )


def calculate_glcm_from_many_images(
    images, distances, angles, levels, symmetric=True, normed=True
):
    return [
        calculate_glcm_from_single_image(
            img, distances, angles, levels, symmetric, normed
        )
        for img in images
    ]


def calculate_glcm_matrix_for_each_category(
    categories, images, distances, angles, levels, symmetric=True, normed=True
):
    return {
        category: calculate_glcm_from_many_images(
            images[category], distances, angles, levels, symmetric, normed
        )
        for category in categories
    }

def extract_glcm_features(glcm, props):
    return np.array([graycoprops(glcm, prop).flatten() for prop in props]).flatten()


def extract_glcm_features_from_many_images(glcm_list, props):
    return [extract_glcm_features(glcm, props) for glcm in glcm_list]


def extract_glcm_features_for_each_category(categories, glcm_list, props):
    return {
        category: extract_glcm_features_from_many_images(glcm_list[category], props)
        for category in categories
    }

# Nova função para paralelizar a extração de características GLCM
def parallel_extract_glcm_features_from_many_images(glcm_list, props, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(extract_glcm_features)(glcm, props) for glcm in glcm_list
    )

def parallel_extract_glcm_features_for_each_category(categories, glcm_list, props, n_jobs=-1):
    return {
        category: parallel_extract_glcm_features_from_many_images(glcm_list[category], props, n_jobs)
        for category in categories
    }

def parallel_calculate_glcm_from_many_images(images, distances, angles, levels, symmetric=True, normed=True, n_jobs=-1):
    return Parallel(n_jobs=n_jobs)(
        delayed(calculate_glcm_from_single_image)(img, distances, angles, levels, symmetric, normed) for img in images
    )

def parallel_calculate_glcm_matrix_for_each_category(categories, images, distances, angles, levels, symmetric=True, normed=True, n_jobs=-1):
    return {
        category: parallel_calculate_glcm_from_many_images(images[category], distances, angles, levels, symmetric, normed, n_jobs)
        for category in categories
    }
