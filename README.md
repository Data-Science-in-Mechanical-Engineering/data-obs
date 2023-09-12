# Data-Driven Observability Analysis for Nonlinear Stochastic Systems

## Installation

We recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), 
as it greatly facilitates installing some libraries for Mac M1 users, such as scipy.

Clone the repo, `cd` into its root directory, and run:
```
conda env create --file environment.yml
```

To activate the created environment:
```
conda activate dis-test
```

### Updating your environment if the YML file has changed

Activate the environment, `cd` into the root directory, and run:
```
conda env update --file environment.yml --prune
```

### Updating the YML file

Activate the environment, `cd` into the root directory, and run:
```
conda env export --from-history | grep -v "^prefix: " > environment.yml
```
The `--from-history` flag restricts the listed packages to the ones you installed manually, not listing the dependencies.
The `grep` part removes the last line of the export, which indicates the prefix of the environment on your machine.

**Make sure that the `environment.yml` file specifies the versions of all packages. If it does not, add the missing ones.**

## Reproducing Results

### Simulations

To reproduce the simulation results of Sections IV.A and IV.B, run the following commands from the root directory:
```sh
python experiments/run.py --system linear --process_noise 0.01 --measurement_noise 0.01 --n_grid 50 --n_traj 30 --T 2 --dt 0.01 --alpha 0.05 --n_jobs 1 --seed 0
python experiments/run.py --system duffing --initial_state 1 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --n_jobs 1 --seed 0
python experiments/run.py --system duffing --initial_state 2 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --n_jobs 1 --seed 0
python experiments/run.py --system duffing --initial_state 2 --process_noise 0.5 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --n_jobs 1 --seed 0
```
Each command will create a new directory in `Results/[Linear,Duffing]`, where the subdirectory depends on the value of the `system` parameter.
The figures `mmd_with_rejected.pdf` in these directories then respectively correspond to Figures 1, 2a, 2b, and 2c in the article, up to cosmetic changes.

For each of these commands, you can specify the number of multiprocessing threads that are spawned for parallelization by varying `n_jobs`. We recommend that you do not exceed the number of cores of your computer, minus 2.
Each of these commands can take significant time to run due to long simulation times.
On an 8-core standard laptop with `n_jobs` set to 6, expect a few minutes for the first command and several hours for the others.

### Additional Results: Hyperparameter Study

Once a simulation is completed, you can easily re-run the computation of the MMD heatmap and the test outcome by adding the `--plot_only [EXP_NAME]` option to the command.
For instance, assuming you have simulation results saved in `Resuls/Duffing/my_custom_simulation`, you can compute the class of indistinguishability with a lower level $\alpha=0.01$ as follows:
```sh
python experiments/run.py --system duffing --initial_state 1 --process_noise 0.01 --measurement_noise 0.01 --n_grid 50 --n_traj 30 --T 2 --dt 0.01 --alpha 0.01 --n_jobs 1 --seed 0 --no_gramian --plot_only my_custom_simulation
```
Importantly, the parameters `initial_state, process_noise, measurement_noise, n_grid, n_traj, T`, and `dt` are unused in this mode and may differ from the ones with which the simulation was performed.
If you remove the flag `--no_gramian` and `system` is set to `duffing`, these parameters become relevant again.

You can also force a specific choice of $\sigma$ instead of relying on the heuristic by adding `--sigma [VALUE]`.

### Hardware Experiment: Furuta Pendulum

To run the experiment:
```sh
python experiments/quanser_qube_hardware_test.py [MEASURE]
```
where `[MEASURE]` is either 1, 2, or 3, representing respectively measuring only $\theta_1$, only $\theta_2$, or both $\theta_1$ and $\theta_2$.
This will run ten tests in a row with the data provided in the repo, using a subset of `Ntraj1`, `Ntraj2` trajectories from 
each experimental dataset. Change `meas_noise_var` directly in the script to modify the amount of measurement noise.

The results are saved in a folder
```sh
Results/QuanserQube/measure12/QQS2_data_Qube_PFM_Ntraj140_Ntraj240_init1_0.0.
0.0._init2_0.70.0.0./nb_test10/exp_0
```
where the title depends on the two reference points, the number of trajectories for each test and the number of tests. 

The result of each test and the mean results over all tests will be printed in the terminal and in `Results/(...)/exp_0/Specifications.txt`. Run
```sh
python experiments/explore_qube_data.py [MEASURE]
```
to evaluate existing results and plot the estimation by the EKF. 
You can modify the lengthscale of the Gaussian kernel and the level of the test by changing the parameters `test_sigma` and `test_alpha` in the script, respectively.


## If you use this repo, please cite:
```
@article{paper,
	author = {Massiani, Pierre-Fran\c{c}ois and Buisson-Fenet, Mona and Solowjow, Friedrich and {Di Meglio}, Florent and Trimpe, Sebastian},
	journal = {arXiv preprint arXiv:2302.11979},
	title = {{Data-Driven Observability Analysis for Nonlinear Stochastic Systems}},
	year = {2023}
}
```
