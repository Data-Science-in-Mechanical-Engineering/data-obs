import os
from typing import Any, Dict

import numpy as np
from scipy.io import loadmat

from kernel_tests.kernel_independence_test import (
    KernelIndependenceTestBootStrappingMatlab,
)
from kernel_tests.utils import MatlabImplementation

from ..helper import get_matlab_test_files


def extract_data_from_mat_file(file_path: str) -> Dict[str, Any]:
    data_dict = loadmat(file_path)

    data_dict_flatten = {
        "X": data_dict["X"],
        "Y": data_dict["Y"],
        "sigma_x": data_dict["params"]["sigx"][0, 0][0, 0],
        "sigma_y": data_dict["params"]["sigy"][0, 0][0, 0],
        "threshold": data_dict["thresh"],
        "test_stat": data_dict["testStat"],
        # Matlab is 1-indexed, but Python ist 0-indexed
        "permutations": data_dict["indices"] - 1,
        "distribution": data_dict["HSICarr"].reshape((-1,)),
        "alpha": data_dict["alpha"],
    }
    return data_dict_flatten


def test_compare_against_matlab_runs() -> None:
    path = os.path.dirname(__file__)
    path = os.path.join(
        path, "../../src/gretton_matlab/kernel_independence_test/"
    )
    files = get_matlab_test_files(path)
    assert len(files) > 0, (
        "Run generateTestData in "
        "src/gretton_matlab/kernel_independence_test"
    )
    for file in files:
        data = extract_data_from_mat_file(file)

        paper_sigma_x = MatlabImplementation.calculate_sigma(data["X"])
        assert np.isclose(
            paper_sigma_x, data["sigma_x"]
        ), "Different kernel size/SigmaX"

        paper_sigma_y = MatlabImplementation.calculate_sigma(data["Y"])
        assert np.isclose(
            paper_sigma_y, data["sigma_y"]
        ), "Different kernel size/SigmaY"

        kernel_test = KernelIndependenceTestBootStrappingMatlab(
            data["X"], data["Y"], data["alpha"],
        )

        kernel_test.perform_test_given_permutations(data["permutations"])
        assert np.all(
            np.isclose(
                kernel_test.distribution_under_null_hypothesis,
                data["distribution"],
            )
        ), "Different Distribution"
        assert np.isclose(
            kernel_test.threshold, data["threshold"]
        ), "Different Threshold"
        assert np.isclose(
            kernel_test.test_stat, data["test_stat"]
        ), "Different test statistic"
