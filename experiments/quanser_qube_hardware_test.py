import copy
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sdeint
import seaborn as sb
from sklearn.preprocessing import StandardScaler

sys.path.append('.')

from kernels_dynamical_systems.custom_kernels import TrajectoryRBFTwoSampleTest
from dynamical_systems.dynamical_systems import QuanserQubeServo2, \
    GaussianInitializer, LinearMeasurement, GaussianNoise, set_seeds, \
    ContinuousTimeEKF, ConstantInitializer, LinearBrownianMotionNoise
from test_utils import print_test_results, print_series_test_results, \
    save_test_data, read_Qube_csv_PFM

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


# Hardware data loading and observability test for the Quanser Qube Servo2
# with different measurement maps (theta, alpha or both)


if __name__ == '__main__':
    MEASURE = int(sys.argv[1])  # in [1, 2, 12]: measure theta, alpha, or both
    PLOT = False
    ROOT_FOLDER = Path(__file__).parent.parent
    load_EKF = False
    nb_tests = 10  # run the test several times in a row

    seed_np = 0
    seed_rng = 42
    set_seeds(seed_np, seed_rng)

    # Parameters
    dim = 4
    dt = 0.01
    init1 = np.array([0., 0., 0., 0.])
    init2 = np.array([0.7, 0., 0., 0.])
    init_var = 0.
    noise_std = 0.
    meas_noise_var = 1e-3
    data_folder = str(ROOT_FOLDER / 'Data' / 'QQS2_data' / 'Qube_PFM')
    solver = sdeint.itoint
    Ntraj1, Ntraj2 = 40, 40  # random subsets of dataset: max 50
    test_alpha = 0.05

    # Create system
    if MEASURE == 1:
        measurement = LinearMeasurement(C=np.array([[1., 0, 0, 0]]))
        dim_obs = 1
        test_sigma = 1300
    elif MEASURE == 2:
        measurement = LinearMeasurement(C=np.array([[0, 1., 0, 0]]))
        dim_obs = 1
        test_sigma = 1300
    elif MEASURE == 12:
        measurement = LinearMeasurement(C=np.array([[1., 0, 0, 0],
                                                    [0., 1, 0, 0]]))
        dim_obs = 2
        test_sigma = 2500
    else:
        raise NotImplementedError(fr'Measure {MEASURE} not implemented.')

    initializer_1 = GaussianInitializer(
        mean=init1, var=init_var * np.eye(dim)
    )
    initializer_2 = GaussianInitializer(
        mean=init2, var=init_var * np.eye(dim)
    )
    noise = LinearBrownianMotionNoise(sigma=noise_std * np.eye(dim))
    meas_noise = GaussianNoise(
        mean=np.zeros(dim_obs),
        var=meas_noise_var * np.eye(dim_obs)
    )

    syst = QuanserQubeServo2(
        state_initializer=initializer_1,
        noise=noise,
        meas=measurement,
        meas_noise=meas_noise
    )

    if not load_EKF:
        # Save results
        data_folder_split = data_folder.rsplit('Data')[-1][1:].replace('/', '_')
        results_folder = str(
            ROOT_FOLDER / 'Results' / 'QuanserQube' / f'measure{MEASURE}' /
            f'{data_folder_split}_Ntraj1{Ntraj1}_Ntraj2{Ntraj2}_init'
            f'1{init1}_init2{init2}' / f'nb_test{nb_tests}'
        ).replace(' ', '').replace('[', '_').replace(']', '')
        i = 0
        while os.path.isdir(os.path.join(results_folder, f"exp_{i}")):
            i += 1
        results_folder = os.path.join(results_folder, f"exp_{i}")
        os.makedirs(results_folder, exist_ok=False)
        specs_file = os.path.join(results_folder, 'Specifications.txt')
        print(f'Results saved in {results_folder}.')
        with open(specs_file, 'w') as f:
            print(sys.argv[0], file=f)
            print('\n', file=f)

        correct_outcomes = 0
        MMD_ratio = []
        # Read data
        start = time.time()
        exps, t_span = read_Qube_csv_PFM(folder=data_folder, dt=dt)
        T = t_span[-1]
        t1, t2 = t_span, t_span
        for nb in range(nb_tests):
            # Shuffle data and pick out Ntraj1, Ntraj2 trajectories
            full_traj1, full_traj2 = copy.deepcopy(exps[0]), copy.deepcopy(
                exps[1])
            np.random.shuffle(full_traj1)
            np.random.shuffle(full_traj2)
            traj1 = full_traj1[:Ntraj1]  # Remapping done before interpolation
            traj2 = full_traj2[:Ntraj2]
            meas1 = syst.get_output_trajectories(N_traj=Ntraj1, T=T, traj=traj1)
            meas2 = syst.get_output_trajectories(N_traj=Ntraj2, T=T, traj=traj2)
            end_data = time.time()
            print(f'Test nb {nb + 1}')
            save_test_data(folder=results_folder, t1=t1, meas1=meas1, t2=t2,
                           meas2=meas2, nb=nb)

            # Plots
            if PLOT:
                plt.plot(t1, traj1[0, :, 0], label=r'$\theta_1$')
                plt.plot(t1, traj2[0, :, 0], label=r'$\theta_2$')
                plt.legend()
                plt.show()
                plt.plot(t1, traj1[0, :, 1], label=r'$\alpha_1$')
                plt.plot(t2, traj2[0, :, 1], label=r'$\alpha_2$')
                plt.legend()
                plt.show()
                for i in range(meas1.shape[-1]):
                    plt.plot(t1, meas1[:, :, i].T, c='blue', label='Dataset 1')
                    plt.plot(t2, meas2[:, :, i].T, c='orange', label='Dataset 2')
                    plt.legend()
                    plt.title(f'Measurement {i + 1}')
                    plt.show()

            # Test
            test_start = time.time()
            scaler = StandardScaler()  # scaler over (2 * T * N, dim_obs)
            whole_meas = np.transpose(np.concatenate(
                (meas1, meas2), axis=1), (1, 0, 2)).reshape(-1, dim_obs)
            scaler = scaler.fit(whole_meas)
            scaled_meas1 = scaler.transform(
                meas1.reshape(-1, dim_obs)).reshape(Ntraj1, -1, dim_obs)
            scaled_meas2 = scaler.transform(
                meas2.reshape(-1, dim_obs)).reshape(Ntraj2, -1, dim_obs)
            test = TrajectoryRBFTwoSampleTest(X=scaled_meas1, Y=scaled_meas2,
                                              alpha=test_alpha,
                                              sigma=test_sigma)
            test.perform_test(verbose=True)
            result = test.is_null_hypothesis_accepted()
            test_end = time.time()
            correct = False
            if MEASURE == 2:
                if result == True:
                    correct_outcomes += 1
                    correct = True
            else:
                if result == False:
                    correct_outcomes += 1
                    correct = True
            MMD_ratio.append([test.test_stat / test.threshold])
            print_test_results(file=specs_file, nb=nb + 1, test=test,
                               correct=correct, data_time=end_data - start,
                               test_time=test_end - test_start)

        # Save results
        print('\n\n\n')
        print(f'{correct_outcomes} correct outcomes out of {nb_tests} tests: '
              f'success rate of {correct_outcomes / nb_tests * 100}%.')
        init_mean1 = np.mean(traj1[:, 0, :], axis=0)
        init_mean2 = np.mean(traj2[:, 0, :], axis=0)
        init_var1 = np.var(traj1[:, 0, :], axis=0)
        init_var2 = np.var(traj2[:, 0, :], axis=0)
        print_series_test_results(file=specs_file,
                                  correct_outcomes=correct_outcomes,
                                  nb_tests=nb_tests, MMD_ratio=MMD_ratio,
                                  var=locals().copy())

    else:
        results_folder = ROOT_FOLDER / 'Results' / 'QuanserQube' / \
                         f'measure{MEASURE}' / \
                         'QQS2_data_Qube_x1_x2_Ntraj125_Ntraj225_init1_0.0.0.0._init2_-1.0.0.0./nb_test10/exp_0'
        T = 9.87
        t1 = np.arange(0, T, dt)
        t2 = np.arange(0, T, dt)
        meas1 = np.load(
            os.path.join(results_folder, 'Test_data/Dataset1_nb9.npy'))
        meas2 = np.load(
            os.path.join(results_folder, 'Test_data/Dataset2_nb9.npy'))
        obs1 = np.load(os.path.join(results_folder, 'EKF_obs1.npy'))
        obs2 = np.load(os.path.join(results_folder, 'EKF_obs2.npy'))
        traj1 = np.load(os.path.join(results_folder, 'EKF_traj1.npy'))
        traj2 = np.load(os.path.join(results_folder, 'EKF_traj2.npy'))

    # Illustrate results with EKF
    init_estim = np.array([0., 0., 0., 0.])
    if MEASURE == 1:
        init_covar = np.array([1e1, 1e1, 1e1, 1e1])
        process_covar = np.array([1e1, 1e2, 1e4, 1e4])
        meas_covar = 1e-2
    elif MEASURE == 2:
        init_covar = np.array([1e1, 1e1, 1e1, 1e1])
        process_covar = np.array([1e2, 1e1, 1e4, 1e4])
        meas_covar = 1e-2
    elif MEASURE == 12:
        init_covar = np.array([1e1, 1e1, 1e1, 1e1])
        process_covar = np.array([1e1, 1e1, 1e4, 1e4])
        meas_covar = 1e-2
    P0 = np.diag(init_covar)
    Q = np.diag(process_covar)
    R = meas_covar * np.eye(dim_obs)
    obs = ContinuousTimeEKF(dim=dim, system=syst, P0=P0, Q=Q, R=R,
                            state_initializer=ConstantInitializer(init_estim))
    meas_EKF1 = meas1  # np.expand_dims(meas1[0], axis=0)
    u1 = np.zeros(
        (meas_EKF1.shape[0], meas_EKF1.shape[1], syst.controller.dim))
    obs1 = obs.get_observations(t=t1, measurements=meas_EKF1, controls=u1)
    obs1 = syst.remap_angles(obs1)  # how to deal with EKF + remapping?
    with open(os.path.join(results_folder, f'EKF_obs1.npy'), "wb") as f:
        np.save(f, obs1)
    with open(os.path.join(results_folder, f'EKF_traj1.npy'), "wb") as f:
        np.save(f, traj1)
    for i in range(obs1.shape[-1]):
        plt.plot(t1, traj1[0, :, i], label='True')
        plt.plot(t1, obs1[0, :, i], label='EKF')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(rf'$x_{i + 1}$')
        plt.savefig(os.path.join(results_folder, f'Traj_EKF1{i}.pdf'),
                    bbox_inches="tight")
        if PLOT:
            plt.show()
        plt.clf()
        plt.close('all')
    meas_EKF2 = meas2  # np.expand_dims(meas2[0], axis=0)
    u2 = np.zeros(
        (meas_EKF2.shape[0], meas_EKF2.shape[1], syst.controller.dim))
    obs2 = obs.get_observations(t=t2, measurements=meas_EKF2, controls=u2)
    obs2 = syst.remap_angles(obs2)  # how to deal with EKF + remapping?
    with open(os.path.join(results_folder, f'EKF_obs2.npy'), "wb") as f:
        np.save(f, obs2)
    with open(os.path.join(results_folder, f'EKF_traj2.npy'), "wb") as f:
        np.save(f, traj2)
    for i in range(obs2.shape[-1]):
        plt.plot(t2, traj2[0, :, i], label='True')
        plt.plot(t2, obs2[0, :, i], label='EKF')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(rf'$x_{i + 1}$')
        plt.savefig(os.path.join(results_folder, f'Traj_EKF2{i}.pdf'),
                    bbox_inches="tight")
        if PLOT:
            plt.show()
        plt.clf()
        plt.close('all')
    # Pointcloud of EKF results at the end of t_span
    for i in range(obs1.shape[-1] - 1):
        plt.plot(np.sin(obs1[:, -1, i]), np.sin(obs1[:, -1, i + 1]), 'o',
                 label='EKF1')
        plt.plot(np.sin(obs2[:, -1, i]), np.sin(obs2[:, -1, i + 1]), 'o',
                 label='EKF2')
        plt.legend()
        plt.xlabel(rf'$\sin(x_{i + 1})$')
        plt.ylabel(rf'$\sin(x_{i + 2})$')
        plt.savefig(os.path.join(results_folder, f'Ptcld_EKF{i}.pdf'),
                    bbox_inches="tight")
        if PLOT:
            plt.show()
        plt.clf()
        plt.close('all')
    # Box plots of EKF results at the end of t_span
    for i in range(obs1.shape[-1]):
        obs_df = pd.DataFrame({'EKF1': obs1[:, -1, i], 'EKF2': obs2[:, -1, i]})
        if i < 2:
            obs_df = np.sin(obs_df)  # Box plot of sin(angle)
            sb.boxplot(data=obs_df)
            plt.ylabel(rf'$\sin(x_{i + 1})$')
        else:
            sb.boxplot(data=obs_df)
            plt.ylabel(rf'$x_{i + 1}$')
        plt.savefig(os.path.join(results_folder, f'Box_plots_EKF{i}.pdf'),
                    bbox_inches="tight")
        if PLOT:
            plt.show()
        plt.clf()
        plt.close('all')
    # Save EKF parameters
    EKF_specs_file = os.path.join(results_folder, 'EKF_specifications.txt')
    with open(EKF_specs_file, 'w') as f:
        print(f'init_estim : {init_estim}', file=f)
        print('\n', file=f)
        print(f'P0 : {P0}', file=f)
        print('\n', file=f)
        print(f'Q : {Q}', file=f)
        print('\n', file=f)
        print(f'R : {R}', file=f)
        print('\n', file=f)
    print_test_results(file=EKF_specs_file, nb=1, test=test, correct=correct)
