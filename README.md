# Data-Driven Observability Analysis for Nonlinear Stochastic Systems

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

## Cloning

This project uses git submodules.
After cloning the repo, you should run these two comments to fetch the code of the submodules:
```
git submodule init
git submodule update
```

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


## Usage

Move into the repo of source files
```
cd repo/src
```
Then run one of the available experiments, e.g.
```
python experiments/LTI_test.py
```



## To reproduce the results of the paper:

### 1. Linear system with noise:

To run the experiment:
```
python experiments/linear_with_drift.py
```
This creates a folder `Results/Drift/exp_0` and saves the results inside. To 
plot the results, run:
```
python experiments/explore_drift_data.py
```
This will save the resulting plots in `Results/Drift/exp_0`. Change 
`EXPERIMENT_NAME` and other parameters in `experiments/explore_drift_data.py` 
to plots the results of another folder or with other options. 
For instance, change `sigma` to `None` in `experiments/linear_with_drift.py` to 
run the heuristic on `sigma`, then change `PLOT_SIGMAS` to `True` in 
`experiments/explore_drift_data.py` to compute the quantiles of the values 
of `sigma` over the grid as suggested by our meta-heuristic. These quantiles 
are saved in `Results/Drift/exp_0/Specifications.txt` along with all 
important parameters.

### 2. Duffing oscillator:
To run the experiment:
```
python experiments/duffing_heatmaps.py
```
with NEW set to True. Set `GRAMIANS` and `TEST` to True to also compute the 
empirical observability Gramians and run a statistical test between two 
arbitrary points with 1000 trajectories per point. Change `noise_std` and 
`meas_noise_var` to modify the process resp. measurement noise.

The results will be saved in `Results/Duffing/exp_0`; run the same script 
again with `NEW` as `False` and `experiment_folder` as `RESULTS / 'exp_0'` to plot 
existing results. 

Run 
```
python experiments/duffing_plots.py
```
with `exps` a list of folders of existing results to plot them together as in 
Fig. 2 of the paper.

### 3. Furuta pendulum:
To run the experiment:
```
python experiments/quanser_qube_hardware_test.py 1
```
where we input 1 to measure only $\theta_1$, 2 to measure only $\theta_2$, 
and 12 to measure both. This will run ten tests in a row with the data 
provided in the repo, using a subset of `Ntraj1`, `Ntraj2` trajectories from 
each experimental dataset. Change `meas_noise_var` to add measurement noise.

This will create a folder 
```
Results/QuanserQube/measure12/QQS2_data_Qube_PFM_Ntraj140_Ntraj240_init1_0.0.
0.0._init2_0.70.0.0./nb_test10/exp_0
```
for the results, where the title depends on the two reference points, the 
number of trajectories for each test and the number of tests. 

The result of each test and the mean results over all tests will be printed in 
the terminal and in `Results/...
/exp_0/Specifications.txt`. Run
```
python experiments/explore_qube_data.py 1
```
to evaluate existing results and plot the estimation by the EKF. In these 
scripts, `test_sigma` is the lengthscale of the Gaussian kernel, while 
`test_alpha` is the confidence level alpha of the test (sometimes called 
`sigma` respectively `alpha` in other scripts if name is not ambiguous).