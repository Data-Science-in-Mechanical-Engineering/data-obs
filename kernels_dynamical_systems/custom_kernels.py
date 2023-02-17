import numpy as np
from kernel_tests import Kernel, KernelTwoSampleTestBootStrapping, utils, MaximumMeanDiscrepancyBiased


#####################
## Utils Functions ##
#####################

def _preprocess_args(X, Y):
    """
    Returns X and Y as np.ndarrays of shape (N, T, D1, ..., DP), where:
        N is the number of data points (possibly different for X and Y)
        T is the time dimension
        D1, ..., DP are the dimensions of the data
    This function essentially expands 1D arrays and checks for compatibility of the dimensions
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1, 1)
    assert X.shape[1:] == Y.shape[1:]
    return X, Y


def space_time_distance(
    X: np.ndarray,
    Y: np.ndarray,
    space_ord: int | float | str | None = 2,
    time_ord: int | float | str | None = 2,
) -> np.ndarray:
    """
    Inputs:
        X: np.ndarray of shape (N, T, D). The second dimension is intepreted as time, and the third is space.
        Y: np.ndarray of shape (M, T, D). The second dimension is intepreted as time, and the third is space.
        space_ord: parameter as accepted for `ord` in the function np.linalg.norm
        time_ord: parameter as accepted for `ord` in the function np.linalg.norm
    Returns:
        distances: np.ndarray of shape (N, M): the distance matrix
    """
    X, Y = _preprocess_args(X, Y)
    # The subtraction involves the shapes (N, 1, ...) - (1, M, ...), where the "..." are identical
    # NumPy's broadcasting rules return an array of shape (N, M, ...)
    differences = X[:, np.newaxis, ...] - Y[np.newaxis, :, ...]
    pointwise_distances = np.linalg.norm(differences, ord=space_ord, axis=-1)
    distances = np.linalg.norm(pointwise_distances, ord=time_ord, axis=-1)
    return distances


def space_time_l2(
    X: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Inputs:
        X: np.ndarray of shape (N, T, D1, ..., DP). The second dimension is intepreted as time, and the third is space.
        Y: np.ndarray of shape (M, T, D1, ..., DP). The second dimension is intepreted as time, and the third is space.
    Returns:
        distances: np.ndarray of shape (N, M): the distance matrix, pointwise squared
    """
    return space_time_distance(X, Y, space_ord=2, time_ord=2)**2


def get_trajectory_rbf(gamma: float):
    """
    Input:
        gamma: float: the (positive) multiplier of the distance in the exponential
    Returns:
        the rbf function
    """
    def trajectory_rbf(
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """
        Inputs:
            X: np.ndarray of shape (N, T, D). The second dimension is intepreted as time, and the third is space.
            Y: np.ndarray of shape (M, T, D). The second dimension is intepreted as time, and the third is space.
            gamma: float: the (positive) multiplier of the distance in the exponential
        Returns:
            kernel: np.ndarray of shape (N, M): the kernel matrix
        """
        distances = space_time_l2(X, Y)
        kernel = np.exp(-gamma * distances)
        return kernel
    return trajectory_rbf


def calculate_sigma(X: np.ndarray, Y: np.ndarray = None, limit_n_points: int | None = None) -> float:
    if limit_n_points is not None:
        X = X[:limit_n_points, ...]
        if Y is not None:
            Y = Y[:limit_n_points, ...]

    if Y is not None:
        Z = np.vstack([X, Y])
    else:
        Z = X
    distances = np.triu(space_time_l2(Z, Z))
    squared_distances = distances ** 2
    median_distance = np.median(squared_distances[squared_distances > 0])
    return np.sqrt(0.5 * median_distance)


#######################
## Kernel Definition ##
#######################

class TrajectoryRBFKernel(Kernel):
    def __init__(self, sigma):
        kernel_function = get_trajectory_rbf(sigma)
        super().__init__(kernel_function=kernel_function)

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> float:
        if Y is None:
            Y = X
        kernel_value = self.kernel_function(X, Y)
        return kernel_value


#####################
## Two-Sample Test ##
#####################

class TrajectoryRBFTwoSampleTest(KernelTwoSampleTestBootStrapping):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 0.05,
        sigma: float | None = None,
    ):
        if sigma is None:
            sigma = calculate_sigma(X, Y)
        self.sigma = sigma
        gamma = utils.convert_sigma_to_gamma(sigma)
        kernel = TrajectoryRBFKernel(gamma)
        # TODO
        # The function calculate_squared_maximum_mean_discrepancy_matlab multiplies
        # the output of calculate_squared_maximum_mean_discrepancy by X.shape[0].
        # Check why with Claas. Meanwhile, we use the standard method, without multiplication
        test_function = (
            MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy
        )
        super().__init__(X, Y, kernel, test_function, alpha)

    def calculate_threshold(self) -> float:
        return utils.MatlabImplementation.calculate_threshold(
            self.alpha, self.distribution_under_null_hypothesis
        )


if __name__ == '__main__':
    n1 = 90
    n2 = 95
    m = 100
    T = 3
    DIMENSIONS = 2
    ALPHA = 0.05

    sigma2X = np.eye(DIMENSIONS)
    muX = np.zeros(DIMENSIONS)

    sigma2Y = np.eye(DIMENSIONS)
    muY = np.ones(DIMENSIONS)

    X1 = np.random.multivariate_normal(mean=muX, cov=sigma2X, size=(n1, T))
    X2 = np.random.multivariate_normal(mean=muX, cov=sigma2X, size=(n2, T))
    Y1 = np.random.multivariate_normal(mean=muY, cov=sigma2Y, size=(m, T))

    print("Kernel Two-Sample Test")
    print("Different distributions")
    kernel_test = TrajectoryRBFTwoSampleTest(X1, Y1, ALPHA)
    kernel_test.perform_test(verbose=True)
    assert not kernel_test.is_null_hypothesis_accepted()

    print("\nSame distribution")
    kernel_test2 = TrajectoryRBFTwoSampleTest(X1, X2, ALPHA)
    kernel_test2.perform_test(verbose=True)
    assert kernel_test2.is_null_hypothesis_accepted()
