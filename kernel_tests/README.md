This is a fork of [Claas Thesing's repository](https://git.rwth-aachen.de/clathe/kernel_tests.git)

## Kernel-Test
This repository contains a Python implementation for the following two
kernel tests developed by Gretton et al.
- [Kernel-Two sample test](https://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm)
- [Statistial Test for Independence](https://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm)

---
So far, only the bootstrapping approach has been implemented.
The Python implementation is roughly tested against the original MATLAB
implementation, but the difference heuristic is slightly adapted and not
limited to the first 100 data points anymore.
The adapted MATLAB files can also be found in this repository.

## Setup
```Conda``` can be used to setup the environment. A ```environment.yml```
is, therefore, given. For further detail see [conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).


### Example
For a toy example how to run the test, see ``src/toy_example.py``.

### Testing
If you like to test the Python implementation against the MATLAB
implementation yourself, you first have to run the
MATLAB file ``generateTestData.m`` in the corresponding directory, either
``src/gretton_matlab/kernel_two_sample_test`` or
``src/gretton_matlab/kernel_independence_test``.

The test can then be executed with ``pytest``.
