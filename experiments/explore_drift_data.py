import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from drift_utils import plot_mmd

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
plt.rcdefaults()
# sns.set_style('whitegrid')
# Set seaborn colors
sns.set_palette("muted")
# # For manuscript
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
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16})

# Script to reuse data from drift system and make new plots

def get_number_of_closest_other(others, allocation, reference):
    diff = others - reference
    distances = np.linalg.norm(diff, ord=2, axis=-1)
    closest_index = np.argmin(distances)
    number_of_closest = allocation[closest_index]
    return number_of_closest, closest_index


def plot_init_and_other(init_ts, other_ts, other_initial_state):
    ndim = init_ts.shape[-1]
    if ndim == 1:
        title_base = r'$y$'
    else:
        title_base = r'$x_{}$'
    title_base += r' ' + \
        str(np.round(other_initial_state.squeeze(), 3).tolist())
    colors = ['b', 'r']
    for d in range(ndim):
        plt.figure()
        plt.plot(t, init_ts[:, :, d].T, color=colors[d], alpha=0.1)
        plt.plot(t, other_ts[:, :, d].T, color=colors[d])
        title = title_base if ndim == 1 else title_base.format(d + 1)
        plt.title(title)


def get_other_trajs_and_meas(others, allocation, reference):
    n_other, index_other = get_number_of_closest_other(
        others, allocation, reference)
    other_trajs = np.load(experiment_data / TRAJ_PATTERN.format(n_other))
    other_meas = np.load(experiment_data / MEAS_PATTERN.format(n_other))
    return other_trajs, other_meas, index_other


if __name__ == "__main__":
    PLOT_RESULTS = True
    PLOT_DATA = False
    PLOT_SIGMAS = True

    ROOT = Path(__file__).parent.parent
    RESULTS = ROOT / 'Results' / 'Drift'
    EXPERIMENT_NAME = 'exp_0'
    TRAJ_PATTERN = 'nb{}_trajs.npy'
    MEAS_PATTERN = 'nb{}_meas.npy'
    experiment_folder = RESULTS / EXPERIMENT_NAME
    experiment_data = experiment_folder / 'data'
    ref = np.array([1.5, 0.5])  # reference point

    rejected_file = experiment_folder / 'rejected.npy'
    thresholds_file = experiment_folder / 'thresholds.npy'
    mmds_file = experiment_folder / 'mmds.npy'
    others_file = experiment_folder / 'others.npy'
    allocation_file = experiment_folder / 'test_numbers_allocation.npy'

    if rejected_file.exists():
        rejected = np.load(rejected_file)
    else:
        rejected = np.logical_not(np.load(experiment_folder / 'accepted.npy'))
    thresholds = np.load(thresholds_file)
    mmds = np.load(mmds_file)
    others = np.load(others_file)
    allocation = np.load(allocation_file).astype(int)

    N_grid = int(np.round(np.sqrt(others.shape[0])))
    dim = others.shape[-1]

    n_init = '0_INIT'
    init_trajs = np.load(experiment_data / TRAJ_PATTERN.format(n_init))
    init_meas = np.load(experiment_data / MEAS_PATTERN.format(n_init))
    T = 2
    t = np.linspace(0, T, init_trajs.shape[1])
    other_references = [
        np.array([0., 0.]),
        np.array([0.5, -0.]),
        np.array([0., 0.5]),
        np.array([1., 1.])
    ]

    if PLOT_RESULTS:
        m = others.min()
        M = others.max()
        extent = (m, M, m, M)
        fig_file = experiment_folder / 'mmds.pdf'
        plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent)
        fig_file = experiment_folder / 'mmds_with_rejected.pdf'
        plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
                 contour=rejected)
        plt.figure()
        plt.imshow(rejected, extent=extent, origin='lower')
        plt.title('Null Hypothesis Rejected')
        plt.savefig(os.path.join(experiment_folder, f'rejected.pdf'),
                    bbox_inches="tight")
        plt.figure()
        # plt.imshow(thresholds, extent=extent, origin='lower')
        plot_mmd(fig_file=os.path.join(experiment_folder, f'thresholds.pdf'), 
                 mmds=thresholds, ref=ref, extent=extent)
        plt.figure()
        plt.imshow(mmds, extent=extent, origin='lower')
        plt.title('MMD')

        mmd_slice_index = np.argmin(
            np.abs(others.reshape(mmds.shape + (2,))
                   [0, :, 0] - 0.)
        )
        # mmd_slice_index = 25
        plt.figure()
        x = np.linspace(m, M, mmds.shape[0])
        plt.plot(x, mmds[mmd_slice_index, :])
        plt.plot(x, thresholds[mmd_slice_index, :])
        plt.title(
            f"MMD Slice at y={others.reshape(N_grid, N_grid, dim)[0, mmd_slice_index, 0]}"
        )
        plt.figure()
        plt.plot(
            x, mmds[mmd_slice_index, :] / thresholds[mmd_slice_index, :]
        )
        # plt.title(f'MMD Slice at y={other_references[-1][0]}')

    if PLOT_SIGMAS:
        m = others.min()
        M = others.max()
        extent = (m, M, m, M)
        sigmas = np.load(experiment_folder / 'sigmas.npy')
        spec_file = experiment_folder / 'Specifications.txt'
        with spec_file.open('a') as f:
            print(f'Sigmas quantiles (0.1, 0.5, 0.9): '
                    f'{np.quantile(sigmas, (0.1, 0.5, 0.9))}', file=f)
        sigmas_fig = experiment_folder / 'sigmas.pdf'
        plt.figure()
        plot_mmd(fig_file=sigmas_fig, mmds=sigmas, ref=ref, extent=extent)

    if PLOT_DATA:
        for reference in other_references:
            other_trajs, other_meas, index_other = get_other_trajs_and_meas(
                others, allocation, reference)
            plot_init_and_other(init_trajs, other_trajs, others[index_other])
            plot_init_and_other(init_meas, other_meas, others[index_other])

    if PLOT_RESULTS or PLOT_DATA or PLOT_SIGMAS:
        plt.show()
