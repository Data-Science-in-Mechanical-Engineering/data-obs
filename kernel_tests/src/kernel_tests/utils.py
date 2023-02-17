import math

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from .kernel import Kernel, convert_sigma_to_gamma


def get_kernel_for_paper_implementation(
    X: np.ndarray, Y: np.ndarray, sigma: float | None
) -> Kernel:
    if sigma is None:
        sigma = calculate_sigma_paper(X, Y)
    kernel = create_rbf_kernel(sigma)
    return kernel


def calculate_sigma_paper(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    distances = np.triu(pairwise_distances(Z))
    median_distance = np.median(distances[distances > 0])
    return median_distance


def create_rbf_kernel(sigma: float) -> Kernel:
    gamma = convert_sigma_to_gamma(sigma)
    kernel = Kernel(gamma=gamma)
    return kernel


class MatlabImplementation:
    @staticmethod
    def calculate_threshold(alpha: float, distribution: np.ndarray) -> float:
        # use floor(x+0.5) to circumvent Python round() issues
        # round(1.5) = 2 = round(2.5)
        index = math.floor((1 - alpha) * len(distribution) + 0.5)
        return distribution[index - 1]

    @staticmethod
    def get_kernel(
        X: np.ndarray, Y: np.ndarray = None, sigma: float | None = None
    ) -> Kernel:
        if sigma is None:
            sigma = MatlabImplementation.calculate_sigma(X, Y)
        kernel = create_rbf_kernel(sigma)
        return kernel

    @staticmethod
    def calculate_sigma(X: np.ndarray, Y: np.ndarray = None) -> float:
        if Y is not None:
            Z = np.vstack([X, Y])
        else:
            Z = X
        distances = np.triu(pairwise_distances(Z))
        squared_distances = distances ** 2
        median_distance = np.median(squared_distances[squared_distances > 0])
        return np.sqrt(0.5 * median_distance)
