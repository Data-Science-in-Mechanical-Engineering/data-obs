from typing import Callable

import numpy as np

from ..boot_strap_test import BootStrapTest
from ..boot_strapper import BootStrapper
from ..kernel import Kernel
from ..utils import get_kernel_for_paper_implementation
from .kernel_two_sample_test import KernelTwoSampleTest
from .maximum_mean_discrepancy import MaximumMeanDiscrepancyBiased


class KernelTwoSampleTestBootStrapping(KernelTwoSampleTest, BootStrapTest):
    @classmethod
    def with_rbf_kernel(
        cls, X: np.ndarray, Y: np.ndarray, alpha: float = 0.05
    ):
        return cls.with_rbf_kernel_sigma(X, Y, None, alpha)

    @classmethod
    def with_rbf_kernel_sigma(
        cls, X: np.ndarray, Y: np.ndarray, sigma: float, alpha: float = 0.05
    ):
        kernel = get_kernel_for_paper_implementation(X, Y, sigma)
        return cls(X, Y, kernel, alpha=alpha)

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_function: Kernel,
        test_function: Callable = MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy,  # pylint: disable=line-too-long
        alpha: float = 0.05,
    ):
        # pylint: disable=too-many-arguments
        KernelTwoSampleTest.__init__(self, X, Y, kernel_function, alpha)
        boot_strapper = BootStrapper(
            test_function,
            self.m,
            self.n,
            self.kernel_matrix,
        )
        # alpha gets overriden, because of diamond inheritance
        BootStrapTest.__init__(self, alpha, boot_strapper)
