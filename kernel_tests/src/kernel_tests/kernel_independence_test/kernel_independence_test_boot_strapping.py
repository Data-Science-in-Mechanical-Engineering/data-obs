from typing import Callable

import numpy as np

from ..boot_strap_test import BootStrapTest
from ..boot_strapper import IndependenceBootStrapper
from ..kernel import Kernel
from ..utils import get_kernel_for_paper_implementation
from .kernel_independence_test import KernelIndependenceTest
from .maximum_mean_discrepancy import MaximumMeanDiscrepancyBiased


class KernelIndependenceTestBootStrapping(
    KernelIndependenceTest, BootStrapTest
):
    @classmethod
    def with_rbf_kernel(
        cls, X: np.ndarray, Y: np.ndarray, alpha: float = 0.05
    ):
        return cls.with_rbf_kernel_sigma(X, Y, None, alpha)

    @classmethod
    def with_rbf_kernel_sigma(
        cls, X: np.ndarray, Y: np.ndarray, sigma: float, alpha: float = 0.05
    ):
        kernel_x = get_kernel_for_paper_implementation(X, Y, sigma)
        kernel_y = get_kernel_for_paper_implementation(X, Y, sigma)
        return cls(X, Y, kernel_x, kernel_y, alpha=alpha)

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_function_x: Kernel,
        kernel_function_y: Kernel,
        alpha: float = 0.05,
        # pylint: disable=line-too-long
        test_function: Callable = MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy_given_hkh,
    ):
        # pylint: disable=too-many-arguments
        KernelIndependenceTest.__init__(
            self, X, Y, kernel_function_x, kernel_function_y, alpha
        )

        matrix_h = np.eye(self.m) - 1 / self.m * np.ones((self.m, self.m))
        matrix_hlh = matrix_h @ self.kernel_matrix_x @ matrix_h
        boot_strapper = IndependenceBootStrapper(
            test_function, self.m, self.kernel_matrix_y, matrix_hlh
        )
        BootStrapTest.__init__(self, alpha, boot_strapper)
