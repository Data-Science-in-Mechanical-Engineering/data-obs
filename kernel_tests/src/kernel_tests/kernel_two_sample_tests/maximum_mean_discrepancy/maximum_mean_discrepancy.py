from abc import ABC, abstractmethod


class MaximumMeanDiscrepancy(ABC):
    @staticmethod
    @abstractmethod
    def calculate_squared_maximum_mean_discrepancy(m, n, kernel_matrix):
        """
        Notice the squared.
        """
