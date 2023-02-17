import numpy as np

from .maximum_mean_discrepancy import MaximumMeanDiscrepancy


class MaximumMeanDiscrepancyBiased(MaximumMeanDiscrepancy):
    @staticmethod
    def calculate_squared_maximum_mean_discrepancy(m, n, kernel_matrix):
        x_kernel_matrix = np.sum(kernel_matrix[:m, :m])
        y_kernel_matrix = np.sum(kernel_matrix[m:, m:])
        xy_kernel_matrix = np.sum(kernel_matrix[:m, m:])
        # fmt: off
        squared_maximum_mean_discrepancy = (
            1/m**2 * x_kernel_matrix
            - 2/(m*n) * xy_kernel_matrix
            + 1/n**2 * y_kernel_matrix
        )
        # fmt: on
        return squared_maximum_mean_discrepancy

    @staticmethod
    def calculate_squared_maximum_mean_discrepancy_matlab(m, n, kernel_matrix):
        return (
            m
            * MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy(
                m, n, kernel_matrix
            )
        )
