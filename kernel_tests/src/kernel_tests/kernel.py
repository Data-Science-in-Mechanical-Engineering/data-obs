from typing import Callable

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels


def convert_sigma_to_gamma(sigma: float) -> float:
    # fmt: off
    return 1 / (2 * sigma**2)
    # fmt: on


class Kernel:
    def __init__(self, kernel_function: str | Callable = "rbf", **kernel_args):
        self.kernel_function = kernel_function
        self.kernel_args = kernel_args

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> float:
        kernel_value = pairwise_kernels(
            X, Y, metric=self.kernel_function, **self.kernel_args
        )
        return kernel_value
