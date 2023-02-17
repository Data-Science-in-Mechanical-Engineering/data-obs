from abc import ABC, abstractmethod


class MaximumMeanDiscrepancy(ABC):
    @staticmethod
    @abstractmethod
    def calculate_squared_maximum_mean_discrepancy(
        kernel_matrix_x, kernel_matrix_y
    ):
        """
        Notice the squared.
        """
