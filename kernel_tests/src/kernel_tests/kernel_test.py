from abc import ABC, abstractmethod

import numpy as np

from .kernel import Kernel


class Test(ABC):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.threshold = None
        self.test_stat = None

    @abstractmethod
    def perform_test(
        self, verbose: bool = False, random_state: int | None = None
    ) -> None:
        # pylint: disable=fixme
        # TODO: move random_state in subclasses if it is not used by further
        #  tests (check matlab files)
        pass

    @abstractmethod
    def calculate_threshold(self) -> float:
        pass

    def write_result_to_std_out(self) -> None:
        print("Threshold: ", self.threshold)
        print("MMD: ", self.test_stat)
        if self.is_null_hypothesis_accepted():
            print("Nullhypothese accepted")
        else:
            print("Nullhypothese rejected")

    def is_null_hypothesis_accepted(self) -> bool:
        try:
            return self.test_stat <= self.threshold
        except TypeError as type_error:
            raise RuntimeError(
                "Test is not run before evaluating. Run test "
                "via test.perform_test()"
            ) from type_error


class KernelTest(Test, ABC):
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel_function: Kernel,
        alpha: float = 0.05,
    ):
        Test.__init__(self, alpha)
        self.X = X
        self.Y = Y
        self._init_kernel_matrix(kernel_function)

    @abstractmethod
    def _init_kernel_matrix(self, kernel_function: Kernel) -> None:
        self.kernel_matrix = None
