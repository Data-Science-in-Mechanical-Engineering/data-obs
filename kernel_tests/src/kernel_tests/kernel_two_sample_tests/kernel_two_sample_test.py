from abc import ABC

import numpy as np

from ..kernel import Kernel
from ..kernel_test import KernelTest


class KernelTwoSampleTest(KernelTest, ABC):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_function: Kernel,
        alpha: float = 0.05,
    ):
        KernelTest.__init__(self, X, Y, kernel_function, alpha)
        self.m = X.shape[0]
        self.n = Y.shape[0]

    def _init_kernel_matrix(self, kernel_function: Kernel) -> None:
        xy = np.vstack([self.X, self.Y])
        self.kernel_matrix = kernel_function(xy)
