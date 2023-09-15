# Data-Driven Observability Analysis for Nonlinear Stochastic Systems

This repository accompanies the article [Data-Driven Observability Analysis for Nonlinear Stochastic Systems](https://arxiv.org/abs/2302.11979).

**Disclaimer**: the given commands and scripts assume that you have a UNIX-based operating system, e.g. Ubuntu or MacOS.
If you are running Windows, you will most likely need to adapt the commands.


## Installation
Clone the repository.
Create a virtual environment, move to the project's root directory, and run 
```sh
pip install -r requirements.txt
```
This project was developed using Python 3.10 and was not tested with other versions.

## Reproducing Results

### Simulations

To reproduce the simulation results of Sections IV.A and IV.B, run the following command from the project's root directory and after having activated the virtual environment:
```sh
scripts/reproduce_article.sh
```
Importantly, before running this script, you should modify the variable `N_JOBS` on line 5 to match the number of multiprocessing threads you want the computations to spawn.
We recommend that you do not exceed the number of cores of your computer, minus 2.
Depending on your OS configuration, you may need to grant the script executing permissions, e.g. using `chmod u+x scripts/reproduce_article.sh`.

This script will create a folder `Results/Article`, where you will find the different figures.
Re-running the script will not re-run an experiment if the corresponding folder exists in `Results/Article`.
Furthermore, you should ensure that the folders `Results/{Linear,Duffing}/exp_0` do not exist before running `reproduce_article.sh`.

Finally, in order to reproduce the exact Figure 2, you should run 
```sh
export PYTHONPATH=$PYTHONPATH:. & python experiments/plot_Fig2.py
```
**after** having run `scripts/reproduce_article.sh`.
Note that the legends will still look different if you do not have `pdflatex` installed.

### Hyperparameter Study

You can run a hyperparameter study by executing the following command:
```sh
scripts/reproduce_rebuttal.sh
```
Here again, you should specify the variable `N_JOBS` on line 5 of the script, grant executing permissions, and ensure that the folders `Results/{Linear,Duffing}/exp_0` do not exist prior to running the script.
This script will store its results in `Results/Rebuttal`.

#### Interpretation
Figure R1 shows what happens when the choice of $\sigma$ is not made uniformly across $x_\mathrm{b}$: every $x_\mathrm{b}$ has a different $\sigma$, which is computed using the heuristic from [1].
We see that the values of the MMD are no longer comparable across different $x_\mathrm{b}$, contrary to Figures 1 and 2 in the article.

Figure R2 shows the influence of the parameter $T$.
The specific values are available in the file `Specifications.txt`. We see that changing $T$ only mildly affects the relative ordering of the MMDs and the test's outcome, justifying the lack of hyperparameter study in the article.

Figure R3 shows how $\alpha$ influences the test's outcome.
Increasing $\alpha$ shrinks the set of points identified as indistinguishable (red pixels).
This is expected from the definition, as the points that are _not_ in red are the ones where the test triggers and $\alpha$ is the test's false positive rate.
Additionally, $\alpha$ does not influence the MMD values.

#### Further Simulations

You can perform additional simulations by running 
```sh
python experiments/run.py [PARAMETERS]
```
with appropriate parameters.
We refer you to the scripts `scripts/{reproduce_article.sh,reproduce_rebuttal.sh}` to see what the available parameters are.
You can also run 
```sh
python experiments/run.py --help
```
for documentation.

If you do this, you may need to add the project's root directory to your `PYTHONPATH` prior to running `run.py`.
This is achieved by running the following command from the project's root:
```sh
export PYTHONPATH=$PYTHONPATH:.
```

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

## References

[1] Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Sch ̈olkopf, and Alexander Smola. A
Kernel Two-Sample Test. Journal of Machine Learning Research, 13(25):723–773, 2012