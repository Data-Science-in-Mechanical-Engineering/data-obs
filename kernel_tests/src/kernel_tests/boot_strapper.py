from typing import Callable

import numpy as np


class BootStrapper:
    def __init__(
        self,
        boot_strap_func: Callable,
        n_samples_x: int,
        n_samples_y: int,
        kernel_matrix: np.ndarray,
    ):
        self.boot_strap_func = boot_strap_func
        self.n_samples_x = n_samples_x
        self.n_samples_y = n_samples_y
        self.kernel_matrix = kernel_matrix

    def boot_strap(
        self, n_iterations: int, random_state: int | None = None
    ) -> np.ndarray:
        permutation_generator = np.random.default_rng(random_state)
        test_stats = np.zeros(n_iterations)
        total_samples = self.n_samples_x + self.n_samples_y
        for i in range(n_iterations):
            permutation = permutation_generator.permutation(total_samples)
            test_stats[i] = self._calculate_permuted_statistic(permutation)
        return np.sort(test_stats)

    def boot_strap_given_permutations(
        self, permutations: np.ndarray
    ) -> np.ndarray:
        test_stats = np.zeros(len(permutations))
        for i, permutation in enumerate(permutations):
            test_stats[i] = self._calculate_permuted_statistic(permutation)
        return np.sort(test_stats)

    def boot_strap_without_permutation(self) -> np.ndarray:
        return self.boot_strap_given_permutations(
            np.arange(self.n_samples_x + self.n_samples_y).reshape((1, -1))
        )

    def _calculate_permuted_statistic(self, permutation: np.ndarray) -> float:
        permuted_kernel = self.kernel_matrix[permutation, permutation[:, None]]
        return self.boot_strap_func(
            self.n_samples_x, self.n_samples_y, permuted_kernel
        )


class IndependenceBootStrapper(BootStrapper):
    def __init__(
        self,
        boot_strap_func: Callable,
        n_samples: int,
        kernel_matrix_k: np.ndarray,
        kernel_matrix_l: np.ndarray,
    ):
        assert kernel_matrix_k.shape == kernel_matrix_l.shape
        assert n_samples == kernel_matrix_k.shape[0]
        super().__init__(
            boot_strap_func, n_samples, n_samples, kernel_matrix_k
        )
        self.kernel_matrix_l = kernel_matrix_l

    def boot_strap(
        self, n_iterations: int, random_state: int | None = None
    ) -> np.ndarray:
        permutation_generator = np.random.default_rng(random_state)
        test_stats = np.zeros(n_iterations)
        for i in range(n_iterations):
            permutation = permutation_generator.permutation(self.n_samples_x)
            test_stats[i] = self._calculate_permuted_statistic(permutation)
        return np.sort(test_stats)

    def boot_strap_without_permutation(self) -> np.ndarray:
        return self.boot_strap_given_permutations(
            np.arange(self.n_samples_x).reshape((1, -1))
        )

    def _calculate_permuted_statistic(self, permutation: np.ndarray) -> float:
        permuted_kernel = self.kernel_matrix[permutation, permutation[:, None]]
        return self.boot_strap_func(permuted_kernel, self.kernel_matrix_l)
