import numpy as np

from ..utils import MatlabImplementation
from .kernel_independence_test_boot_strapping import (
    KernelIndependenceTestBootStrapping,
)


class KernelIndependenceTestBootStrappingMatlab(
    KernelIndependenceTestBootStrapping
):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 0.05,
        sigma_x: float | None = None,
        sigma_y: float | None = None,
    ):
        # pylint: disable=too-many-arguments
        kernel_x = MatlabImplementation.get_kernel(X, sigma_x)
        kernel_y = MatlabImplementation.get_kernel(Y, sigma_y)
        super().__init__(X, Y, kernel_x, kernel_y, alpha)

    def calculate_threshold(self) -> float:
        return MatlabImplementation.calculate_threshold(
            self.alpha, self.distribution_under_null_hypothesis
        )
