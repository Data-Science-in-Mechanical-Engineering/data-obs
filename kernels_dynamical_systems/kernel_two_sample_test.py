import numpy as np


def _preprocess_args(x, y):
    # Trajectories should have the same length and dimension
    assert x.shape[1:] == y.shape[1:]
    if x.ndim == 1:
        x = x.reshape(-1, 1, 1)
        y = y.reshape(-1, 1, 1)
    if x.ndim == 2:
        x = x[:, :, np.newaxis]
        y = y[:, :, np.newaxis]
    return x, y


def linear_kernel(X, Y):
    return X.T @ Y


def rbf_kernel(x, y, sigma=1, **kwargs):
    x, y = _preprocess_args(x, y)
    e = x[:, np.newaxis, ...] - y[np.newaxis, :, ...]
    axes_for_norm = tuple(range(2, e.ndim))
    # axes_for_norm = None
    d_old = np.linalg.norm(e, axis=axes_for_norm)

    pointwise_distances = np.linalg.norm(e, ord=2, axis=-1)
    d = np.linalg.norm(pointwise_distances, ord=2, axis=-1)

    return np.exp(- d**2 / sigma**2 / 2)


def median_distance(x1, x2, default=np.inf):
    x1, x2 = _preprocess_args(x1, x2)
    # Use at most 100 points for computational efficiency
    m1 = min(100, x1.shape[0])
    m2 = min(100, x2.shape[0])
    if m1 == 0 or m2 == 0:
        return default

    e = x1[:m1, np.newaxis, ...] - x2[np.newaxis, :m2, ...]
    axes_for_norm = tuple(range(2, e.ndim))
    d = np.linalg.norm(e, axis=axes_for_norm)

    # x1 = x1[:m, :]
    # x2 = x2[:m, :]
    # d = np.linalg.norm(x1 - x2, axis=1)

    med = np.median(d)

    return med


def mmd2_estimate(x, y, kernel, **kernel_kwargs):
    k_xx = kernel(x, x, **kernel_kwargs)
    k_yy = kernel(y, y, **kernel_kwargs)
    k_xy = kernel(x, y, **kernel_kwargs)

    # If x.shape[0] (resp. y.shape[0]) is 0, then the terms in 1/n (resp. 1/m) are 0
    # in the final sum. So we set n (resp. m) to 1 to avoid numerical issues
    n = max(x.shape[0], 1)
    m = max(y.shape[0], 1)

    estimate = k_xx.sum() / n**2 + k_yy.sum() / m**2 - k_xy.sum() * 2 / (n*m)

    return estimate


if __name__ == '__main__':
    x = np.arange(12).reshape(3, 2, 2)
    out = rbf_kernel(x, x, bandwidth=10)
    print(out)
