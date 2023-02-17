import numpy as np

from kernel_tests.kernel_independence_test import (
    KernelIndependenceTestBootStrappingMatlab,
)
from kernel_tests.kernel_two_sample_tests import (
    KernelTwoSampleTestBootStrappingMatlab,
)

if __name__ == "__main__":
    m = 100
    n = 100
    DIMENSIONS = 2
    ALPHA = 0.05

    sigma2X = np.eye(DIMENSIONS)
    muX = np.zeros(DIMENSIONS)

    sigma2Y = np.eye(DIMENSIONS)
    muY = np.ones(DIMENSIONS)

    X = np.random.multivariate_normal(mean=muX, cov=sigma2X, size=m)
    Y = np.random.multivariate_normal(mean=muY, cov=sigma2Y, size=n)

    print("Kernel Two-Sample Test")
    print("Different distribution - Different Sigma!")
    kernel_test = KernelTwoSampleTestBootStrappingMatlab(X, Y, ALPHA)
    kernel_test.perform_test(verbose=True)
    assert not kernel_test.is_null_hypothesis_accepted()

    print("\nSame distribution")
    kernel_test2 = KernelTwoSampleTestBootStrappingMatlab(X, X, ALPHA)
    kernel_test2.perform_test(verbose=True)
    assert kernel_test2.is_null_hypothesis_accepted()

    print()
    print("--------------")
    print()

    print("Kernel Independence Test")
    print("Independent")
    independence_test = KernelIndependenceTestBootStrappingMatlab(X, Y, ALPHA)
    independence_test.perform_test(verbose=True)
    assert independence_test.is_null_hypothesis_accepted()

    print("\nNot independent - same data")
    independence_test2 = KernelIndependenceTestBootStrappingMatlab(X, X, ALPHA)
    independence_test2.perform_test(verbose=True)
    assert not independence_test2.is_null_hypothesis_accepted()
