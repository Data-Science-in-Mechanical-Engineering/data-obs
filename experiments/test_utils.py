import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from dynamical_systems.dynamical_systems import interpolate_trajectory, \
    QuanserQubeServo2


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



# Utility functions for the experiments scripts



def print_test_results(file, nb, test, correct, data_time=None,
                       test_time=None, var=None):
    result = test.is_null_hypothesis_accepted()
    with open(file, 'a') as f:
        print(f'Results of test nb {nb}:', file=f)
        print('p-value:', test.p_value, file=f)
        print('alpha:', test.alpha, file=f)
        print('sigma:', test.sigma, file=f)
        print('Threshold: ', test.threshold, file=f)
        print('MMD: ', test.test_stat, file=f)
        print(f'The initial states are '
              f'{"distinguishable" if result == False else "indistinguishable"}',
              file=f)
        if correct:
            print(f'The outcome is CORRECT', file=f)
        else:
            print(f'The outcome is INCORRECT', file=f)
        if data_time is not None:
            print(fr'Time to generate data: {data_time}', file=f)
        if test_time is not None:
            print(fr'Time for test: {test_time}', file=f)
        print('\n', file=f)
        if var is not None:
            for key, val in var.items():
                print(key, ': ', val, file=f)


def print_series_test_results(file, correct_outcomes, nb_tests,
                              MMD_ratio=None, var=None):
    with open(file, 'a') as f:
        print('\n\n', file=f)
        print(f'{correct_outcomes} correct outcomes out of {nb_tests} tests: '
              f'success rate of {correct_outcomes / nb_tests * 100}%.', file=f)
        if MMD_ratio is not None:
            print(
                f'Mean ratio MMD / threshold: '
                f'{np.mean(np.array(MMD_ratio))}', file=f)
        print('\n\n\n', file=f)
        if var is not None:
            for key, val in var.items():
                print(key, ': ', val, file=f)


def save_test_data(folder, t1, meas1, t2, meas2, nb=0):
    test_folder = os.path.join(folder, 'Test_data')
    os.makedirs(test_folder, exist_ok=True)
    f1 = os.path.join(test_folder, f'Dataset1_nb{nb}.npy')
    f2 = os.path.join(test_folder, f'Dataset2_nb{nb}.npy')
    with open(f1, "wb") as f:
        np.save(f, meas1)
    with open(f2, "wb") as f:
        np.save(f, meas2)
    for i in range(meas1.shape[-1]):
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        plt.plot(meas1[0, :, i], c=colors[0])
        plt.plot(meas2[0, :, i], c=colors[1])
        for j in range(1, len(meas1)):
            plt.plot(meas1[j, :, i], c=colors[0])
        for j in range(1, len(meas1)):
            plt.plot(meas2[j, :, i], c=colors[1])
        plt.legend(labels=['Dataset 1', 'Dataset 2'])
        # plt.plot(t1, meas1[:, :, i].T, c='blue')
        # plt.plot(t2, meas2[:, :, i].T, c='orange')
        # plt.title(f'Measurements {nb + 1}')
        plt.xlabel('Time')
        plt.ylabel(rf'$y$')
        plt.savefig(os.path.join(test_folder, f'Meas{i}_nb{nb}.pdf'),
                    bbox_inches="tight")
        plt.clf()
        plt.close('all')


def read_Qube_csv(folder, dt, T0=0., nb_files=2):
    # Read hardware data from Qube:
    # For each initial condition, cut out individual trajectory from long csv
    # Find min T from all trajectories, create time vector(T0,T,dt)
    # Then interpolate each trajectory on time vector (after restarting at T0)
    # Return list of datasets
    # TODO more efficient?
    T = 120.
    exps = []
    cuts = []
    exps_cut = []
    # init = []
    for i in range(nb_files):
        # Read individual trajectories, find idx to cut and max T
        exp = np.genfromtxt(
            os.path.join(folder, f'x{i + 1}.csv'), delimiter=',')[1:, 1:]
        cut_idx = [0]
        # inits = [exp[0, 1]]
        for j in range(1, len(exp)):
            if j == len(exp) - 1:
                if exp[j, -1] - exp[cut_idx[-1], -1] < T:
                    T = exp[j - 1, -1] - exp[cut_idx[-1], -1]
                cut_idx.append(j)
            elif exp[j, -1] < exp[j - 1, -1]:
                if exp[j - 1, -1] - exp[cut_idx[-1], -1] < T:
                    T = exp[j - 1, -1] - exp[cut_idx[-1], -1]
                cut_idx.append(j)
                # inits.append(exp[j, 1])
        cuts.append(cut_idx)
        exps.append(exp)
        # init.append(inits)
    for i in range(nb_files):
        # Interpolate individual trajectories over same time vector
        cut_idx = cuts[i]
        exp = exps[i]
        time = np.arange(start=T0, stop=T, step=dt)  # time vector (0, T)
        exp_cut = np.zeros((len(cut_idx) - 1, len(time), exp.shape[-1] - 1))
        for k in range(len(cut_idx) - 1):
            trajt = exp[cut_idx[k]:cut_idx[k + 1]]
            traj, t = trajt[:, :-1], trajt[:, -1] - trajt[0, -1] + T0  # restart
            x_func = interpolate_trajectory(t=t, meas_traj=traj)
            exp_cut[k] = x_func(time)
        exps_cut.append(exp_cut)
    # Delete some experiments
    # Dataset 1: too close to mean, nb to roughly match dataset 2
    # Dataset 2: nb 13 actually starts at theta=0, others too far from mean
    # del_lis0 = []
    # del_list0 = [0, 1, 2, 3, 7, 8, 10, 11, 14, 20, 25, 26, 28]
    # del_list0 = [4, 6, 7, 12, 20, 39, 45]  # < 10 * var from mean
    del_list0 = [1, 2, 4, 6, 7, 9, 10, 12, 13, 15, 18, 19, 20, 23, 28, 29,
                 31, 34, 35, 39, 41, 45, 47]  # < 50 * var from mean
    exps_cut[0] = np.delete(exps_cut[0], del_list0, axis=0)
    # del_list1 = [13]  # outlier: theta0 = 0
    # del_list1 = [5, 6, 7, 12, 13, 16, 17, 18, 21, 23, 28, 29, 40, 42, 45,
    #              47]  # > 2 * var from mean
    # del_list1 = [5, 6, 7, 12, 13, 14, 16, 17, 18, 21, 23, 25, 28, 29, 35, 40,
    #              42, 43, 45, 47, 49]  # > 1.6 * var from mean
    del_list1 = [5, 6, 7, 12, 13, 14, 16, 17, 18, 20, 21, 23, 25, 28, 29, 30,
                 34, 35, 40, 42, 43, 44, 45, 47, 49]  # > 1.5 * var from mean
    exps_cut[1] = np.delete(exps_cut[1], del_list1, axis=0)
    return exps_cut, time


def read_Qube_csv_PFM(folder, dt, init=[0, 45]):
    # Read hardware data from Qube:
    # T0 determined as first time angular velocities reach local min
    # Find T0 for each trajectory and min total duration T of all experiments
    # Create time vector (T0, T0+T, dt)
    # Then interpolate each trajectory on time vector and restart at 0
    # Return list of datasets
    # TODO more efficient?
    nb_init = len(init)
    T = 120.
    exps = []
    T0 = []
    j = 0
    for i in range(nb_init):
        exps.append([])
        T0.append([])
        while os.path.exists(
                os.path.join(folder, f'init_{init[i]}_data_{j}.csv')):
            if j in [4]:  # Ignore a few weird experiments
                j += 1
                continue
            # Read individual trajectories, find min T0 and max T
            exp = np.genfromtxt(os.path.join(
                folder, f'init_{init[i]}_data_{j}.csv'), delimiter=',')[1:, 1:]

            init_velo = np.linalg.norm(exp[0, 2:4])
            for t in range(1, len(exp)):
                # T0 defined as the time of first local min of velocities
                angle_velo = np.linalg.norm(exp[t, 2:4])
                if angle_velo > init_velo:
                    T0[i].append(exp[t - 1, -1])
                    break
                init_velo = angle_velo
            if exp[-1, -1] - T0[i][-1] < T:
                T = exp[-1, -1] - T0[i][-1]
            j += 1
            exps[i].append(exp)
    exps_interp = []
    for i in range(nb_init):
        time = np.arange(start=T0[i][0], stop=T0[i][0] + T, step=dt)
        exp_interp = np.zeros(
            (len(exps[i]), len(time), exp.shape[-1] - 1))
        for j in range(len(exps[i])):
            # Interpolate individual trajectories over time vector
            time = np.arange(start=T0[i][j], stop=T0[i][j]+T, step=dt)
            exp = exps[i][j]
            traj, t = exp[:, :-1], exp[:, -1]
            # if j in [4]:  # Correct a few weird experiments
            #     traj[:, 0] -= np.pi
            # Remap traj before interpolation, else outliers at discontinuities
            traj = QuanserQubeServo2.remap_hardware_angles(traj, mod=True,
                                                           add_pi_alpha=False)
            x_func = interpolate_trajectory(t=t, meas_traj=traj)
            exp_interp[j] = x_func(time)
        exps_interp.append(exp_interp)
    # Adjust initial distributions
    init_mean1, init_mean2 = np.mean(exps_interp[0][:, 0], axis=0), \
                             np.mean(exps_interp[1][:, 0], axis=0)
    init_var1, init_var2 = np.var(exps_interp[0][:, 0], axis=0), \
                           np.var(exps_interp[1][:, 0], axis=0)
    print('Initial distributions before preprocessing:')
    print(init_mean1, init_var1)
    print(init_mean2, init_var2)
    for k in range(exp.shape[-1] - 1):
        plt.plot(exps_interp[0][:, 0, k], 'x')
        plt.plot(exps_interp[1][:, 0, k], 'x')
        plt.savefig(os.path.join(folder, f'inits_{k}_before.pdf'))
        plt.clf()
        plt.close('all')
    # Delete some experiments
    # Dataset 1:
    # Dataset 2:
    del_list0 = []  # < 50 * var from mean
    exps_interp[0] = np.delete(exps_interp[0], del_list0, axis=0)
    del_list1 = []  # > 1.5 * var from mean
    exps_interp[1] = np.delete(exps_interp[1], del_list1, axis=0)
    # Result
    init_mean1, init_mean2 = np.mean(exps_interp[0][:, 0], axis=0), \
                             np.mean(exps_interp[1][:, 0], axis=0)
    init_var1, init_var2 = np.var(exps_interp[0][:, 0], axis=0), \
                           np.var(exps_interp[1][:, 0], axis=0)
    print('Initial distributions after preprocessing:')
    print(init_mean1, init_var1)
    print(init_mean2, init_var2)
    for k in range(exp.shape[-1] - 1):
        plt.plot(exps_interp[0][:, 0, k], 'x')
        plt.plot(exps_interp[1][:, 0, k], 'x')
        plt.savefig(os.path.join(folder, f'inits_{k}_after.pdf'))
        plt.clf()
        plt.close('all')
    return exps_interp, time - time[0]  # restart at 0
