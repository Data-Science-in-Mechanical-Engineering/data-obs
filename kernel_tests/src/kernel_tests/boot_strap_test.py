from abc import ABC

import numpy as np

from .boot_strapper import BootStrapper
from .kernel_test import Test


class BootStrapTest(Test, ABC):
    def __init__(self, alpha: float, boot_strapper: BootStrapper):
        super().__init__(alpha)
        self.distribution_under_null_hypothesis = None
        self.p_value = None
        self.boot_strapper = boot_strapper

    def perform_test(
        self, verbose: bool = False, random_state: int | None = None
    ):
        self.perform_test_for_n_iterations(
            n_iterations=1000, random_state=random_state, verbose=verbose
        )

    def perform_test_for_n_iterations(
        self,
        n_iterations: int,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.distribution_under_null_hypothesis = (
            self.boot_strapper.boot_strap(n_iterations, random_state)
        )
        self._perform_test(verbose=verbose)

    def perform_test_given_permutations(
        self, permutations: np.ndarray
    ) -> None:
        self.distribution_under_null_hypothesis = (
            self.boot_strapper.boot_strap_given_permutations(permutations)
        )
        self._perform_test()

    def _perform_test(self, verbose: bool = False) -> None:
        self.threshold = self.calculate_threshold()
        self.test_stat = self.boot_strapper.boot_strap_without_permutation()[0]
        self.p_value = self.calculate_p_value()
        if verbose:
            self.write_result_to_std_out()

    def calculate_threshold(self) -> float:
        return np.quantile(
            self.distribution_under_null_hypothesis, 1 - self.alpha
        )

    def calculate_p_value(self) -> float:
        return (
            self.distribution_under_null_hypothesis > self.test_stat
        ).mean()

    def write_result_to_std_out(self) -> None:
        print("p-value:", self.p_value)
        super().write_result_to_std_out()
