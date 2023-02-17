from abc import ABC

import numpy as np

from ..kernel import Kernel
from ..kernel_test import KernelTest


class KernelIndependenceTest(KernelTest, ABC):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_function_x: Kernel,
        kernel_function_y: Kernel,
        alpha: float = 0.05,
    ):
        assert X.shape == Y.shape, "X and Y must be of the same shape"
        # call with two times X such that self.kernel_matrix is
        # kernel_function(X, X)
        super().__init__(X, Y, kernel_function_x, alpha)
        self.kernel_matrix_y = kernel_function_y(Y)
        self.m = X.shape[0]

    @property
    def kernel_matrix_x(self) -> np.ndarray:
        return self.kernel_matrix

    @kernel_matrix_x.setter
    def kernel_matrix_x(self, matrix) -> None:
        self.kernel_matrix = matrix

    @kernel_matrix_x.deleter
    def kernel_matrix_x(self) -> None:
        del self.kernel_matrix

    def _init_kernel_matrix(self, kernel_function) -> None:
        self.kernel_matrix_x = kernel_function(self.X)
