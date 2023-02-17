import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import sys
sys.path.append('.')

from kernels_dynamical_systems import TrajectoryRBFTwoSampleTest
from dynamical_systems.dynamical_systems import LTISystem, GaussianNoise, \
    GaussianInitializer, LinearMeasurement, set_seeds
from kernels_dynamical_systems.kernel_two_sample_test import mmd2_estimate, \
    rbf_kernel
from test_utils import save_test_data, print_test_results

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
plt.rcdefaults()
sb.set_style('whitegrid')
# Set seaborn colors
sb.set_palette("muted")
# For manuscript
plot_params = {
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.serif': 'Palatino',
    'font.size': 16,
    "pgf.preamble": "\n".join([
        r'\usepackage{bm}',
    ]),
    'text.latex.preamble': [r'\usepackage{amsmath}',
                            r'\usepackage{amssymb}',
                            r'\usepackage{cmbright}'],
}
plt.rcParams.update(plot_params)
# # Previously
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
# plt.rc('font', family='serif')
# plt.rcParams.update({'font.size': 16})

# Script to apply the K2S test on measured trajectories of two LTI systems:
# one observable, one not observable (both discrete-time)
# System 1 : Euler discretization of harmonic oscillator
# System 2 from: https://www.ece.rutgers.edu/~gajic/psfiles/chap5traCO.pdf

if __name__ == '__main__':
    PLOT = True
    ROOT_FOLDER = Path(__file__).parent.parent

    seed_np = 0
    seed_rng = 42
    set_seeds(seed_np, seed_rng)

    COMPARE_MMD_WITH_OLD_IMPLEMENTATION = False

    # Observable LTI system: Euler discretization of harmonic oscillator, y = x1
    dim = 2
    dim_obs = 1
    dt = 1e-2
    meas_var = 1e-1
    process_var = meas_var * np.sqrt(dt)
    A = np.array([[1., dt], [-dt, 1.]])
    B = np.array([[0.], [0.]])
    C = np.array([[1., 0.]])
    D = None
    T = 30
    init11 = np.array([0., 1.])
    init12 = np.array([0., 2.])
    init_var = 1e-3
    initializer_1 = GaussianInitializer(
        mean=init11, var=np.diag([init_var, init_var]))
    initializer_2 = GaussianInitializer(
        mean=init12, var=np.diag([init_var, init_var]))
    sys1 = LTISystem(
        dim=dim, A=A, B=B,
        state_initializer=initializer_1,
        noise=GaussianNoise(mean=np.zeros(dim), var=process_var*np.eye(dim)),
        meas=LinearMeasurement(C=C, D=D),
        meas_noise=GaussianNoise(mean=np.zeros(
            dim_obs), var=meas_var*np.eye(dim_obs))
    )
    # Create 2 datasets from 2 initial conditions
    Ntraj = 100
    obs11 = sys1.get_trajectories(N_traj=Ntraj, T=T)
    meas11 = sys1.get_output_trajectories(N_traj=Ntraj, T=T, traj=obs11)

    sys1.state_initializer = initializer_2
    obs12 = sys1.get_trajectories(N_traj=Ntraj, T=T)
    meas12 = sys1.get_output_trajectories(N_traj=Ntraj, T=T, traj=obs12)
    if PLOT:
        plt.plot(obs11[0, :, 0], label=r'$x^1$')
        plt.plot(obs11[0, :, 1], label=r'$\dot{x}^1$')
        plt.plot(obs12[0, :, 0], label=r'$x^2$')
        plt.plot(obs12[0, :, 1], label=r'$\dot{x}^2$')
        plt.legend()
        plt.show()
        # plt.plot(meas11[:Ntraj, :, 0].T, c='blue', label=r'$y^1$')
        # plt.plot(meas12[:Ntraj, :, 0].T, c='orange', label=r'$y^2$')
        # plt.legend()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        plt.plot(meas11[0, :, 0], c=colors[0])
        plt.plot(meas12[0, :, 0], c=colors[1])
        for j in range(1, len(meas11)):
            plt.plot(meas11[j, :, 0], c=colors[0])
        for j in range(1, len(meas12)):
            plt.plot(meas12[j, :, 0], c=colors[1])
        plt.legend(labels=['Dataset 1', 'Dataset 2'])
        plt.xlabel('Time')
        plt.ylabel(r'$y$')
        plt.show()
    # K2S test
    scaler1 = StandardScaler()  # scaler over (2 * T * N, dim_obs)
    whole_meas1 = np.transpose(np.concatenate(
        (meas11, meas12), axis=1), (1, 0, 2)).reshape(-1, dim_obs)
    scaler1 = scaler1.fit(whole_meas1)
    meas11 = scaler1.transform(
        meas11.reshape(-1, dim_obs)).reshape(-1, T, dim_obs)
    meas12 = scaler1.transform(
        meas12.reshape(-1, dim_obs)).reshape(-1, T, dim_obs)
    test1 = TrajectoryRBFTwoSampleTest(X=meas11, Y=meas12, alpha=0.05)
    print("Results:")
    test1.perform_test(verbose=True)
    result1 = test1.is_null_hypothesis_accepted()
    correct1 = True if result1 == False else False
    print(f'The initial states are '
          f'{"distinguishable" if result1 == False else "indistinguishable"}')
    print(f'The outcome is {"CORRECT" if correct1 == True else "INCORRECT"}')
    # Save results
    results_folder1 = str(
        ROOT_FOLDER / 'Results' / 'LTI_test1' /
        f'Ntraj{Ntraj}_T{T}_init1{init11}_init2{init12}'
    ).replace(' ', '').replace('[', '_').replace(']', '_')
    i = 0
    while os.path.isdir(os.path.join(results_folder1, f"exp_{i}")):
        i += 1
    results_folder1 = os.path.join(results_folder1, f"exp_{i}")
    os.makedirs(results_folder1, exist_ok=False)
    specs_file1 = os.path.join(results_folder1, 'Specifications.txt')
    print(f'Results saved in {results_folder1}.')
    t_span = np.arange(start=0., stop=T)
    save_test_data(folder=results_folder1, t1=t_span, meas1=meas11,
                   t2=t_span, meas2=meas12)
    print_test_results(file=specs_file1, nb=1, test=test1, correct=correct1,
                       var=locals().copy())

    if COMPARE_MMD_WITH_OLD_IMPLEMENTATION:
        mmd1 = mmd2_estimate(meas11, meas12, rbf_kernel, sigma=test1.sigma)
        print(f'MMD (old implementation): {mmd1}')
        is_close1 = np.isclose(test1.test_stat, mmd1, rtol=1e-6, atol=1e-6)
        print(f"The values are {'CLOSE' if is_close1 else 'DIFFERENT'}")

    # Non observable LTI system
    dim = 2
    dim_obs = 1
    meas_var = 1e-1
    process_var = meas_var * np.sqrt(dt)
    init_var = 1e-3
    A = np.array([[1., -2.], [-3., -4.]]) * 0.2
    B = np.array([[0.], [0.]])
    C = np.array([[1., 2.]])
    D = None
    init21 = np.array([0., 1.])
    init22 = np.array([2., 0.])  # np.array([1., 1.])  # distinguishable!
    sys2 = LTISystem(
        dim=dim, A=A, B=B,
        state_initializer=GaussianInitializer(
            mean=init21, var=np.diag([init_var, init_var])
        ),
        noise=GaussianNoise(mean=np.zeros(dim), var=process_var*np.eye(dim)),
        meas=LinearMeasurement(C=C, D=D),
        meas_noise=GaussianNoise(mean=np.zeros(
            dim_obs), var=meas_var*np.eye(dim_obs))
    )

    # Create 2 datasets from 2 initial conditions
    obs21 = sys2.get_trajectories(N_traj=Ntraj, T=T)
    meas21 = sys2.get_output_trajectories(N_traj=Ntraj, T=T, traj=obs21)
    sys2.state_initializer = GaussianInitializer(  # distinguishable!
        mean=init22, var=np.diag([init_var, init_var]))
    obs22 = sys2.get_trajectories(N_traj=Ntraj, T=T)
    meas22 = sys2.get_output_trajectories(N_traj=Ntraj, T=T, traj=obs22)
    if PLOT:
        plt.plot(obs21[0, :, 0], label=r'$x^1$')
        plt.plot(obs21[0, :, 1], label=r'$\dot{x}^1$')
        plt.plot(obs22[0, :, 0], label=r'$x^2$')
        plt.plot(obs22[0, :, 1], label=r'$\dot{x}^2$')
        plt.legend()
        plt.show()
        # plt.plot(meas21[:Ntraj, :, 0].T, c='blue', label=r'$y^1$')
        # plt.plot(meas22[:Ntraj, :, 0].T, c='orange', label=r'$y^2$')
        # plt.legend()
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        plt.plot(meas11[0, :, 0], c=colors[0])
        plt.plot(meas12[0, :, 0], c=colors[1])
        for j in range(1, len(meas11)):
            plt.plot(meas11[j, :, 0], c=colors[0])
        for j in range(1, len(meas12)):
            plt.plot(meas12[j, :, 0], c=colors[1])
        plt.legend(labels=['Dataset 1', 'Dataset 2'])
        plt.xlabel('Time')
        plt.ylabel(r'$y$')
        plt.show()
    # K2S test
    scaler2 = StandardScaler()  # scaler over (2 * T * N, dim_obs)
    whole_meas2 = np.transpose(np.concatenate(
        (meas21, meas22), axis=1), (1, 0, 2)).reshape(-1, dim_obs)
    scaler2 = scaler2.fit(whole_meas1)
    meas21 = scaler2.transform(
        meas21.reshape(-1, dim_obs)).reshape(-1, T, dim_obs)
    meas22 = scaler2.transform(
        meas22.reshape(-1, dim_obs)).reshape(-1, T, dim_obs)
    test2 = TrajectoryRBFTwoSampleTest(X=meas21, Y=meas22, alpha=0.05)
    print("Results:")
    test2.perform_test(verbose=True)
    result2 = test2.is_null_hypothesis_accepted()
    correct2 = True if result2 == True else False
    print(f'The initial states are '
          f'{"distinguishable" if result2 == False else "indistinguishable"}')
    print(f'The outcome is {"CORRECT" if correct2 == True else "INCORRECT"}')
    # Save results
    results_folder2 = str(
        ROOT_FOLDER / 'Results' / 'LTI_test2' /
        f'Ntraj{Ntraj}_T{T}_init1{init21}_init2{init22}'
    ).replace(' ', '').replace('[', '_').replace(']', '_')
    i = 0
    while os.path.isdir(os.path.join(results_folder2, f"exp_{i}")):
        i += 1
    results_folder2 = os.path.join(results_folder2, f"exp_{i}")
    os.makedirs(results_folder2, exist_ok=False)
    specs_file2 = os.path.join(results_folder2, 'Specifications.txt')
    print(f'Results saved in {results_folder2}.')
    t_span = np.arange(start=0., stop=T)
    save_test_data(folder=results_folder2, t1=t_span, meas1=meas21,
                   t2=t_span, meas2=meas22)
    print_test_results(file=specs_file2, nb=1, test=test2, correct=correct2,
                       var=locals().copy())

    if COMPARE_MMD_WITH_OLD_IMPLEMENTATION:
        mmd2 = mmd2_estimate(meas21, meas22, rbf_kernel, sigma=test2.sigma)
        print(f'MMD (old implementation): {mmd2}')
        is_close2 = np.isclose(test2.test_stat, mmd2, rtol=1e-6, atol=1e-10)
        print(f"The values are {'CLOSE' if is_close2 else 'DIFFERENT'}")
