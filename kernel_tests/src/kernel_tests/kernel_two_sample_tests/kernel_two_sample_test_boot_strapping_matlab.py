import numpy as np

from ..utils import MatlabImplementation
from .kernel_two_sample_test_boot_strapping import (
    KernelTwoSampleTestBootStrapping,
)
from .maximum_mean_discrepancy import MaximumMeanDiscrepancyBiased


class KernelTwoSampleTestBootStrappingMatlab(KernelTwoSampleTestBootStrapping):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 0.05,
        sigma: float | None = None,
    ):
        # pylint: disable=too-many-arguments
        kernel = MatlabImplementation.get_kernel(X, Y, sigma)
        test_function = (
            MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy_matlab
        )
        super().__init__(X, Y, kernel, test_function, alpha)

    def calculate_threshold(self) -> float:
        return MatlabImplementation.calculate_threshold(
            self.alpha, self.distribution_under_null_hypothesis
        )
