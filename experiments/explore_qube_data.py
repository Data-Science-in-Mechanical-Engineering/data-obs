import copy
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sdeint
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from brokenaxes import brokenaxes
# https://github.com/bendichter/brokenaxes

sys.path.append('.')

from kernels_dynamical_systems.custom_kernels import TrajectoryRBFTwoSampleTest
from dynamical_systems.dynamical_systems import QuanserQubeServo2, \
    GaussianInitializer, LinearMeasurement, GaussianNoise, set_seeds, \
    ContinuousTimeEKF, ConstantInitializer, LinearBrownianMotionNoise
from test_utils import print_test_results, print_series_test_results, \
    save_test_data, read_Qube_csv

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
plt.rcdefaults()
# Set seaborn colors
sns.set_palette("muted")
# # For manuscript
# sns.set_style('whitegrid')
# plot_params = {
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
#     'font.serif': 'Palatino',
#     'font.size': 16,
#     "pgf.preamble": "\n".join([
#         r'\usepackage{bm}',
#     ]),
#     'text.latex.preamble': [r'\usepackage{amsmath}',
#                             r'\usepackage{amssymb}',
#                             r'\usepackage{cmbright}'],
# }
# plt.rcParams.update(plot_params)
# Previously
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
# plt.rc('font', family='serif')
# plt.rcParams.update({'font.size': 16})


# Script to reuse data from Qube experiment and make new plots

def phase_portrait_data(meas1, meas2, results_folder, kde=False):
    if not kde:
        # https://matplotlib.org/stable/gallery/color/color_cycle_default.html
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        for i in range(len(meas1)):
            plt.plot(meas1[i, :, 0], meas1[i, :, 1], 'x', c=colors[0])
            plt.plot(meas2[i, :, 0], meas2[i, :, 1], '*', c=colors[1])
        plt.legend(labels=['Dataset 1', 'Dataset 2'])
    else:
        # https://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot
        dim1 = meas1.shape[0]
        dim2 = meas1.shape[1]
        sns.kdeplot(
            x=meas1[:, :, 0].reshape(dim1 * dim2),
            y=meas1[:, :, 1].reshape(dim1 * dim2),
            fill=True,
            clip=(-np.pi, 2 * np.pi),
            cut=0,
            thresh=0.15, levels=150,
            alpha=0.5,
            label='Dataset 1',
        )
        sns.kdeplot(
            x=meas2[:, :, 0].reshape(dim1 * dim2),
            y=meas2[:, :, 1].reshape(dim1 * dim2),
            fill=True,
            clip=(-np.pi, 2 * np.pi),
            cut=0,
            thresh=0.15, levels=150,
            alpha=0.5,
            label='Dataset 2',
        )
    plt.title('Datasets')
    plt.xlabel(r'$\theta_1$')
    plt.ylabel(r'$\theta_2$')
    if not kde:
        plt.savefig(os.path.join(results_folder, f'Dataset.pdf'),
                    bbox_inches="tight")
    else:
        plt.savefig(os.path.join(results_folder, f'Dataset_kde.pdf'),
                    bbox_inches="tight")
    plt.clf()
    plt.close('all')


def EKF_trajs(t1, t2, obs1, obs2, traj1, traj2, results_folder):
    for i in range(obs1.shape[-1]):
        plt.plot(t1, traj1[0, :, i], label='True')
        plt.plot(t1, obs1[0, :, i], label='EKF')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(rf'$x_{i + 1}$')
        plt.savefig(os.path.join(results_folder, f'Traj_EKF1{i}.pdf'),
                    bbox_inches="tight")
        plt.clf()
        plt.close('all')
    for i in range(obs2.shape[-1]):
        plt.plot(t2, traj2[0, :, i], label='True')
        plt.plot(t2, obs2[0, :, i], label='EKF')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(rf'$x_{i + 1}$')
        plt.savefig(os.path.join(results_folder, f'Traj_EKF2{i}.pdf'),
                    bbox_inches="tight")
        plt.clf()
        plt.close('all')


def EKF_pointcloud(obs1, obs2, results_folder):
    for i in range(obs1.shape[-1] - 1):
        plt.plot(np.sin(obs1[:, -1, i]), np.sin(obs1[:, -1, i + 1]), 'o',
                 label='Dataset 1')
        plt.plot(np.sin(obs2[:, -1, i]), np.sin(obs2[:, -1, i + 1]), 'o',
                 label='Dataset 2')
        plt.legend()
        plt.xlabel(rf'$\sin(x_{i + 1})$')
        plt.ylabel(rf'$\sin(x_{i + 2})$')
        plt.savefig(os.path.join(results_folder, f'Ptcld_EKF{i}.pdf'),
                    bbox_inches="tight")
        plt.clf()
        plt.close('all')


def EKF_boxplots(obs1, obs2, results_folder):
    for i in range(obs1.shape[-1]):
        obs_df = pd.DataFrame({'Dataset 1': obs1[:, -1, i],
                               'Dataset 2': obs2[:, -1, i]})
        if i < 2:
            obs_df = np.sin(obs_df)  # Box plot of sin(angle)
            sns.boxplot(data=obs_df)
            plt.ylabel(rf'$\sin(\theta_{i + 1})$')
        else:
            sns.boxplot(data=obs_df)
            plt.ylabel(rf'$x_{i + 1}$')
        plt.savefig(os.path.join(results_folder, f'Box_plots_EKF{i}.pdf'),
                    bbox_inches="tight")
        plt.clf()
        plt.close('all')


def EKF_violinplots(t1, obs1, t2, obs2, results_folder):
    def format_data(time, data, label):
        """
        The label is a string identifying the class of all points in `data`
        Typically, it should be set to 'x' or 'z' and represent the initial state.
        Formats the data to put it in a long-type pd.DataFrame
        The output is a 2D np.ndarray, with N * T rows and three columns, respectively
        containing the timestamp, the value, and the label
        """
        time_formatted = np.tile(time.reshape(-1, 1), (data.shape[0], 1, 1))
        initial_state_formatted = np.full_like(
            time_formatted, label, dtype=np.object_)
        data = np.dstack(
            (time_formatted, data, initial_state_formatted))
        return data

    # Data to plot is sin(theta_1): only keep this
    data_0 = format_data(t1, np.sin(obs1[..., 0]), 'Dataset 1').reshape(-1, 3)
    data_1 = format_data(t2, np.sin(obs2[..., 0]), 'Dataset 2').reshape(-1, 3)
    data = np.vstack((data_0, data_1))

    # The dataframe has three columns (time, measurement, label), and
    # (N+M)*T rows
    # To change the name of the column "measurement", change the corresponding line here
    df = pd.DataFrame(dict(
        time=data[:, 0].astype(float),
        measurement=data[:, 1].astype(float),
        label=data[:, 2].astype(str)
    ))

    # With manual broken axis
    # https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
    f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')
    ax.set_xlim(0, 4)
    ax2.set_xlim(4, 9)
    ssub1 = [np.floor(0.1 * i * 10) / 10 for i in range(1, 5)]
    # sub1 = np.array([0.01] + ssub1)  # cannot be finer than dt=0.01
    sub1 = np.array(ssub1)
    df1 = df.loc[df.time.isin(sub1)]
    sub2 = np.array([np.floor(2 * i + 1) for i in range(5)])
    df2 = df.loc[df.time.isin(sub2)]
    sns.violinplot(
        ax=ax,
        data=df1,
        # change "measurement" here in accordance with the changes in df
        y=r"measurement", x=r"time", hue="label",
        split=True,
        inner=None,
        linewidth=0,
    )
    sns.violinplot(
        ax=ax2,
        data=df2,
        # change "measurement" here in accordance with the changes in df
        y=r"measurement", x=r"time", hue="label",
        split=True,
        inner=None,
        linewidth=0,
    )
    # Set the axes names/ticks/labels manually
    ax.spines['right'].set_visible(False)
    # ax.xaxis.set_ticklabels([0] + ssub1)
    ax.xaxis.set_ticklabels(sub1)
    ax.xaxis.set_label_text('')
    ax.yaxis.set_label_text('')
    ax.legend('', frameon=False)
    ax2.spines['left'].set_visible(False)
    ax2.xaxis.set_ticklabels(sub2)
    ax2.xaxis.set_label_text('')
    ax2.yaxis.set_label_text('')
    ax2.yaxis.set_ticks_position('none')
    ax2.legend('', frameon=False)
    # Add diagonal signs to broken axes
    d = .015  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    # Final details for whole plot
    f.subplots_adjust(hspace=0.05)
    # plt.legend()
    plt.legend('', frameon=False)
    f.supxlabel('Time')
    f.supylabel(rf'$\sin(\theta_1)$')

    plt.savefig(os.path.join(results_folder, f'Violin_plots_EKF.pdf'),
                bbox_inches="tight")
    plt.clf()
    plt.close('all')


def EKF_RMSE(t1, error_traj1, t2, error_traj2, results_folder, dim=None):
    RMSE1 = np.linalg.norm(error_traj1, axis=-1)
    median1 = np.median(RMSE1, axis=0)
    quant1 = np.quantile(RMSE1, (0.25, 0.75), axis=0)
    RMSE2 = np.linalg.norm(error_traj2, axis=-1)
    median2 = np.median(RMSE2, axis=0)
    quant2 = np.quantile(RMSE2, (0.25, 0.75), axis=0)
    plt.plot(t1, median1, label='Dataset 1')
    plt.fill_between(t1, quant1[0], quant1[1], alpha=0.3)
    plt.plot(t2, median2, label='Dataset 2')
    plt.fill_between(t2, quant2[0], quant2[1], alpha=0.3)
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel(rf'$\| \hat{{x}} - x \|$')
    if dim is not None:
        plt.savefig(os.path.join(results_folder, f'RMSE_EKF{dim}.pdf'),
                    bbox_inches="tight")
    else:
        plt.savefig(os.path.join(results_folder, f'RMSE_EKF.pdf'),
                    bbox_inches="tight")
    plt.clf()
    plt.close('all')


def EKF_RMSE12(t1, obs1, traj1, t2, obs2, traj2, results_folder, dim=None):
    # Error - dataset nb - dim
    error11 = np.expand_dims(obs1[..., 0] - traj1[..., 0], -1)
    error21 = np.expand_dims(obs2[..., 0] - traj2[..., 0], -1)
    error12 = np.expand_dims(obs1[..., 1] - traj1[..., 1], -1)
    error22 = np.expand_dims(obs2[..., 1] - traj2[..., 1], -1)

    # Error on theta_1
    RMSE11 = np.linalg.norm(error11, axis=-1)
    median11 = np.median(RMSE11, axis=0)
    quant11 = np.quantile(RMSE11, (0.25, 0.75), axis=0)
    RMSE21 = np.linalg.norm(error21, axis=-1)
    median21 = np.median(RMSE21, axis=0)
    quant21 = np.quantile(RMSE21, (0.25, 0.75), axis=0)
    # p11, = plt.plot(t1, median11, color='royalblue')
    # plt.fill_between(t1, quant11[0], quant11[1], alpha=0.3, color='royalblue')
    # p21, = plt.plot(t2, median21, color='sandybrown')
    # plt.fill_between(t2, quant21[0], quant21[1], alpha=0.3, color='sandybrown')
    # p11, = plt.plot(t1, median11)
    # plt.fill_between(t1, quant11[0], quant11[1], alpha=0.3)
    # p21, = plt.plot(t2, median21)
    # plt.fill_between(t2, quant21[0], quant21[1], alpha=0.3)

    # Error on theta_2
    RMSE12 = np.linalg.norm(error12, axis=-1)
    median12 = np.median(RMSE12, axis=0)
    quant12 = np.quantile(RMSE12, (0.25, 0.75), axis=0)
    RMSE22 = np.linalg.norm(error22, axis=-1)
    median22 = np.median(RMSE22, axis=0)
    quant22 = np.quantile(RMSE22, (0.25, 0.75), axis=0)
    # p12, = plt.plot(t1, median12, color='slateblue')
    # plt.fill_between(t1, quant12[0], quant12[1], alpha=0.3, color='slateblue')
    # p22, = plt.plot(t2, median22, color='coral')
    # plt.fill_between(t2, quant22[0], quant22[1], alpha=0.3, color='coral')
    p12, = plt.plot(t1, median12)
    plt.fill_between(t1, quant12[0], quant12[1], alpha=0.3)
    p22, = plt.plot(t2, median22)
    plt.fill_between(t2, quant22[0], quant22[1], alpha=0.3)

    plt.yscale('log')
    # Nice legend
    # https://stackoverflow.com/questions/24787041/multiple-titles-in-legend-in-matplotlib
    from matplotlib.patches import Rectangle
    title_proxy = Rectangle((0, 0), 0, 0, color='w')
    # plt.legend([title_proxy, p11, p12, title_proxy, p21, p22],
    #            ['Dataset 1', r'$\theta_1$', r'$\theta_2$',
    #             'Dataset 2', r'$\theta_1$', r'$\theta_2$'], labelspacing=0.3)
    plt.legend([p12, p22], ['Dataset 1', 'Dataset 2'], labelspacing=0.3,
               fontsize='xx-large')
    plt.xlabel('Time')
    # plt.ylabel(rf'$\| \hat{{x}} - x \|$')
    plt.ylabel(rf'Estimation error $\| \hat{{\theta}}_2 - \theta_2 \|$')
    plt.savefig(os.path.join(results_folder, f'RMSE12_EKF.pdf'),
                bbox_inches="tight")
    plt.clf()
    plt.close('all')


if __name__ == '__main__':
    MEASURE = int(sys.argv[1])  # in [1, 2, 12]: measure theta, alpha, or both

    ROOT = Path(__file__).parent.parent
    RESULTS = ROOT / 'Results' / 'QuanserQube'
    results_folder = RESULTS / f'measure{MEASURE}' / \
                     'QQS2_data_Qube_PFM_Ntraj140_Ntraj240_init1_0.0.0.0' \
                     '._init2_0.70.0.0.' / 'nb_test10' / 'Good2'

    seed_np = 0
    seed_rng = 42
    set_seeds(seed_np, seed_rng)

    # Parameters
    dim = 4
    dt = 0.01
    T = 9.75
    init1 = np.array([0., 0., 0., 0.])
    init2 = np.array([0.7, 0., 0., 0.])
    init_var = 0.
    noise_std = 0.
    meas_noise_var = 1e-3
    solver = sdeint.itoint
    Ntraj1, Ntraj2 = 40, 40  # random subsets of dataset: max 49
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

    # Read data
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
    # Remap results
    obs1, obs2 = syst.remap_angles(obs1), syst.remap_angles(obs2)
    traj1, traj2 = syst.remap_angles(traj1), syst.remap_angles(traj2)

    if MEASURE == 12:
        # Phase portrait of measurements
        phase_portrait_data(meas1=meas1, meas2=meas2,
                            results_folder=results_folder)

    # EKF parameters
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

    # Plot EKF results
    EKF_trajs(t1=t1, t2=t2, obs1=obs1, obs2=obs2, traj1=traj1, traj2=traj2,
              results_folder=results_folder)
    # Pointcloud of EKF results at the end of t_span
    EKF_pointcloud(obs1=obs1, obs2=obs2, results_folder=results_folder)
    # Box plots of EKF results at the end of t_span
    EKF_boxplots(obs1=obs1, obs2=obs2, results_folder=results_folder)
    # Violin plots of EKF results
    EKF_violinplots(t1=t1, obs1=obs1, t2=t2, obs2=obs2,
                    results_folder=results_folder)

    # Plot EKF RMSE
    EKF_RMSE(t1=t1, error_traj1=obs1 - traj1, t2=t2, error_traj2=obs1 - traj1,
             results_folder=results_folder)
    for i in range(obs1.shape[-1]):
        error_traj1 = np.expand_dims(obs1[..., i] - traj1[..., i], -1)
        error_traj2 = np.expand_dims(obs2[..., i] - traj2[..., i], -1)
        EKF_RMSE(t1=t1, error_traj1=error_traj1, t2=t2, error_traj2=error_traj2,
                 results_folder=results_folder, dim=i)
    EKF_RMSE12(t1=t1, obs1=obs1, traj1=traj1, t2=t2, obs2=obs2, traj2=traj2,
               results_folder=results_folder)

    # Test distinguishability between outputs of EKF obs1, obs2
    test_start = time.time()
    scaler = StandardScaler()  # scaler over (2 * T * N, dim)
    whole_obs = np.transpose(np.concatenate(
        (obs1, obs2), axis=1), (1, 0, 2)).reshape(-1, dim)
    scaler = scaler.fit(whole_obs)
    scaled_obs1 = scaler.transform(
        obs1.reshape(-1, dim)).reshape(Ntraj1, -1, dim)
    scaled_obs2 = scaler.transform(
        obs2.reshape(-1, dim)).reshape(Ntraj2, -1, dim)
    test = TrajectoryRBFTwoSampleTest(X=scaled_obs1, Y=scaled_obs2,
                                      alpha=test_alpha,
                                      sigma=None)  # test_sigma)
    test.perform_test(verbose=True)
    result = test.is_null_hypothesis_accepted()
    test_end = time.time()
    correct = False
    if MEASURE == 2:
        if result == True:
            correct = True
    else:
        if result == False:
            correct = True
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
    print_test_results(file=EKF_specs_file, nb=1, test=test,
                       correct=correct, test_time=test_end - test_start)
