# Dynamical Systems Implementation

Implementations of various dynamical systems.
Intended for research and personal use.

## Unit Testing

Unit tests are located in the `tests` folder. They are intended to be run using the `unittest` Python module, and not as standalone scripts; you would run into import issues. To run a test, open a shell in the **root directory** of the repository, activate the virtual environment, and run one of the following two commands:
```shell
python -m unittest tests.[NAME OF TEST WITHOUT .py]
python -m unittest tests/[NAME OF TEST WITH .py]
```

To run all tests, use the command:
```shell
python -m unittest discover -p "*_test.py" tests
```

It is also possible to configure your IDE to run the tests; we refer you to the documentation of your specific IDE for this. In this case, you will need to specify the test-naming pattern `*_test.py`.