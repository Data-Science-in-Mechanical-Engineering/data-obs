import numpy as np

from .maximum_mean_discrepancy import MaximumMeanDiscrepancy


class MaximumMeanDiscrepancyBiased(MaximumMeanDiscrepancy):
    @staticmethod
    def calculate_squared_maximum_mean_discrepancy(
        kernel_matrix_x, kernel_matrix_y
    ):
        # https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product
        # See more details for possible transformations.
        # Here we stick to the formula of the paper not to the matlab
        # implementation but they are equal because of the possible
        # transformations.
        assert kernel_matrix_x.shape == kernel_matrix_y.shape, (
            "Matrices " "must be of equal shape"
        )
        n_samples = kernel_matrix_x.shape[0]
        matrix_h = np.eye(n_samples) - 1 / n_samples * np.ones(
            (n_samples, n_samples)
        )
        matrix_hlh = matrix_h @ kernel_matrix_y @ matrix_h
        return MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy_given_hkh(
            kernel_matrix_x, matrix_hlh
        )

    @staticmethod
    def calculate_squared_maximum_mean_discrepancy_given_hkh(
        kernel_matrix_l, matrix_hkh
    ):
        # https://en.wikipedia.org/wiki/Trace_(linear_algebra)#Trace_of_a_product
        # See more details for possible transformations.
        # Here we stick not to the formula of the paper but to the matlab
        # implementation but they are equal because of the possible
        # transformations. However, slightly numerical differences can be
        # detected.
        # tr(HKHL) = tr(LHKH) = tr(KHLH) and H, L, K are symmetric
        assert (
            kernel_matrix_l.shape == matrix_hkh.shape
        ), "Matrices must be of equal shape"
        n_samples = kernel_matrix_l.shape[0]
        return 1 / n_samples ** 2 * np.sum(matrix_hkh * kernel_matrix_l)
