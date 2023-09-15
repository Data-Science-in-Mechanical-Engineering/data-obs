#!/bin/bash
# Should be launched with the virtualenv activated from the project root directory
# This script only works if a new experiment is stored as exp_0 for both the linear and duffing systems

N_JOBS=6

export PYTHONPATH=$PYTHONPATH:.

mkdir ./Results/Rebuttal

if [ -d "./Results/Rebuttal/FigR1" ]; then 
    echo "./Results/Rebuttal/FigR1 found. Skipping simulation of Figure R1"
else
    echo "Computing Figure R1"
    python experiments/run.py --system linear --initial_state 1 --process_noise 0.1 --measurement_noise 0.01 --n_grid 50 --n_traj 30 --T 2 --dt 0.01 --sigma individual --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Linear/exp_0 ./Results/Rebuttal/FigR1
fi

if [ -d "./Results/Rebuttal/FigR2e" ]; then
    echo "./Results/Rebuttal/FigR2e found. Skipping simulation of Figure R2"
else
    echo "Computing Figure R2"
    python experiments/run.py --system linear --initial_state 1 --process_noise 0.1 --measurement_noise 0.01 --n_grid 50 --n_traj 30 --T 5 --dt 0.01 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Linear/exp_0 ./Results/Rebuttal/FigR2b

    python experiments/run.py --system linear --initial_state 1 --process_noise 0.1 --measurement_noise 0.01 --n_grid 50 --n_traj 30 --T 10 --dt 0.01 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Linear/exp_0 ./Results/Rebuttal/FigR2c

    python experiments/run.py --system duffing --initial_state 1 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 3 --dt 0.001 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Duffing/exp_0 ./Results/Rebuttal/FigR2e
fi

if [ -d "./Results/Rebuttal/FigR3c" ]; then 
    echo "./Results/Rebuttal/FigR3c found. Skipping simulation of Figure R3"
else
    echo "Computing Figure R3"
    python experiments/run.py --system duffing --initial_state 1 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --seed 0 --no_gramian --no_trajectory --n_jobs $N_JOBS

    cp -r ./Results/Duffing/exp_0 ./Results/Rebuttal/FigR3b

    python experiments/run.py --system duffing --initial_state 1 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.01 --seed 0 --no_gramian --no_trajectory --n_jobs $N_JOBS --plot_only exp_0

    cp -r ./Results/Duffing/exp_0 ./Results/Rebuttal/FigR3a

    python experiments/run.py --system duffing --initial_state 1 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.1 --seed 0 --no_gramian --no_trajectory --n_jobs $N_JOBS --plot_only exp_0

    mv ./Results/Duffing/exp_0 ./Results/Rebuttal/FigR3c
fi