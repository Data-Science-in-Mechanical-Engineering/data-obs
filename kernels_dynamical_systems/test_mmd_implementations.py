"""
Compare the two implementations of the MMD against one another
"""
import numpy as np
from kernel_tests import MaximumMeanDiscrepancyBiased, Kernel
from kernels_dynamical_systems.custom_kernels import TrajectoryRBFKernel
from kernel_tests.src.kernel_tests.kernel import convert_sigma_to_gamma
from kernels_dynamical_systems.kernel_two_sample_test import mmd2_estimate, rbf_kernel


SEED = 0
RNG = np.random.default_rng(SEED)


def get_MMD_with_RBF_kernel_Claas(X, Y, sigma):
    if X.ndim < 3:
        kernel = Kernel(kernel_function='rbf',
                        gamma=convert_sigma_to_gamma(sigma))
    else:
        kernel = TrajectoryRBFKernel(sigma)
    Z = np.vstack((X, Y))
    kernel_matrix = kernel(Z)
    mmd = MaximumMeanDiscrepancyBiased.calculate_squared_maximum_mean_discrepancy(
        X.shape[0], Y.shape[0], kernel_matrix)
    return mmd


def compare_MMD(X, Y, label_X='X', label_Y='Y', verbose=True, eps=1e-6):
    sig = 2
    mmd_1 = mmd2_estimate(x=X, y=Y, kernel=rbf_kernel, sigma=sig)
    mmd_2 = get_MMD_with_RBF_kernel_Claas(X, Y, sigma=sig)
    is_close = np.isclose(mmd_1, mmd_2, rtol=eps, atol=eps)
    if verbose:
        print(f'=== MMD: {label_X} vs. {label_Y} ===')
        print(f'Method 1   (PFM): {mmd_1}')
        print(f'Method 2 (Claas): {mmd_2}')
        print(f'Values are {"CLOSE" if is_close else "DIFFERENT"}')
    return is_close


def generate_data(N, M, P, T, D):
    shape_X = (N, T, D)
    shape_Y = (M, T, D)
    shape_Z = (P, T, D)
    if D == 1:
        shape_X = shape_X[:-1]
        shape_Y = shape_Y[:-1]
        shape_Z = shape_Z[:-1]

    sigma = RNG.normal(size=(np.prod(shape_Y), np.prod(shape_Y)))
    sigma = np.eye(sigma.shape[0]) + sigma.T @ sigma  # Ensure SPD matrix
    X1 = RNG.multivariate_normal(
        np.zeros(np.prod(shape_Y)), sigma).reshape(shape_Y)[:N, ...]
    X2 = RNG.multivariate_normal(
        np.zeros(np.prod(shape_Y)), sigma).reshape(shape_Y)
    X3 = RNG.multivariate_normal(
        np.zeros(np.prod(shape_Z)), np.eye(np.prod(shape_Z))).reshape(shape_Z)

    return X1, X2, X3


def perform_comparison(X1, X2, X3):
    data = [('X1', X1), ('X2', X2), ('X3', X3)]
    all_close = True
    for i, d1 in enumerate(data):
        for d2 in data[i:]:
            l1, Y1 = d1
            l2, Y2 = d2
            all_close = all_close and compare_MMD(
                X=Y1, Y=Y2, label_X=l1, label_Y=l2, verbose=True)
    return all_close


if __name__ == '__main__':
    N = [3] * 3
    M = [2] * 3  # M must be <= N
    P = [5] * 3
    T = [1, 11, 11]
    D = [1, 1, 13]
    all_close = True
    for i in range(len(N)):
        print(
            f'======== Configuration {i+1} (T = {T[i]}, D = {D[i]}) ========')
        X1, X2, X3 = generate_data(N[i], M[i], P[i], T[i], D[i])
        all_close = all_close and perform_comparison(X1, X2, X3)
    print(f"All values as close: {all_close}")
