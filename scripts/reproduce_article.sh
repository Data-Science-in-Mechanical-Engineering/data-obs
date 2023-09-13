#!/bin/bash
# Should be launched with the virtualenv activated from the project root directory
# This script only works if a new experiment is stored as exp_0 for both the linear and duffing systems

N_JOBS=6

export PYTHONPATH=$PYTHONPATH:.

mkdir ./Results/Article

if [ -d "./Results/Article/Fig1" ]; then
    echo "./Results/Article/Fig1 found. Skipping simulation of Figure 1"
else
    echo "Computing Figure 1"
    python experiments/run.py --system linear --initial_state 1 --process_noise 0.01 --measurement_noise 0.01 --n_grid 50 --n_traj 30 --T 2 --dt 0.01 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Linear/exp_0 ./Results/Article/Fig1
fi

if [ -d "./Results/Article/Fig2a" ]; then
    echo "./Results/Article/Fig2a found. Skipping simulation of Figure 2a"
else
    echo "Computing Figure 2a"
    python experiments/run.py --system duffing --initial_state 1 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Duffing/exp_0 ./Results/Article/Fig2a
fi

if [ -d "./Results/Article/Fig2b" ]; then
    echo "./Results/Article/Fig2b found. Skipping simulation of Figure 2b"
else
    echo "Computing Figure 2b"
    python experiments/run.py --system duffing --initial_state 2 --process_noise 0.05 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Duffing/exp_0 ./Results/Article/Fig2b
fi

if [ -d "./Results/Article/Fig2c" ]; then
    echo "./Results/Article/Fig2c found. Skipping simulation of Figure 2c"
else
    echo "Computing Figure 2c"
    python experiments/run.py --system duffing --initial_state 2 --process_noise 0.5 --measurement_noise 0.5 --n_grid 100 --n_traj 50 --T 1 --dt 0.001 --alpha 0.05 --seed 0 --n_jobs $N_JOBS

    mv ./Results/Duffing/exp_0 ./Results/Article/Fig2c
fi


