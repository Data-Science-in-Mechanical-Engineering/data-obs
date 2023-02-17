import numpy as np
from pathlib import Path

import os
import sys
import seaborn as sns
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import scipy
import numpy as np

sys.path.append('.')

from dynamical_systems.dynamical_systems import set_seeds, Duffing, \
    Measurement, GaussianNoise, ConstantInitializer, \
    LinearBrownianMotionNoise, NoController, NoNoise
from drift_utils import get_new_experiment_folder, dump_specs, log_mmds
from linear_with_drift import get_mmd_map, generate_trajs_from_initial_state

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


# Data generation and MMD heatmaps for the Duffing oscillator: pick a
# reference point, draw heatmap of MMD values on a grid w.r.t this point


class DuffingMeasurement(Measurement):
    def __init__(self, alpha, beta, *args, **kwargs):
        dim = 1
        super().__init__(dim=dim, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def jacobian(self, state, t):
        return self.C

    def get_measurement(self, state, *args, **kwargs):
        y = self.alpha / 2 * state[..., 0] ** 2 + state[..., 1] ** 2 / 2 + \
            self.beta / 4 * state[..., 0] ** 4
        return np.expand_dims(y, -1)


def plot_many_mmd(ax, fig_file, mmds, ref=None, extent=None, add_traj=None,
                  add_vect=None, contour=None, alpha=1, log=True,
                  colorbar=False, legend=False):
    # ax = plt.subplot()
    cmap = sns.color_palette("crest_r", as_cmap=True)

    if extent is not None:
        if log:
            im = ax.imshow(
                mmds,
                origin='lower',
                extent=extent,
                cmap=cmap,
                norm=matplotlib.colors.LogNorm(vmin=mmds.min(),
                                               vmax=mmds.max()),
                rasterized=True
            )
        else:
            im = ax.imshow(
                mmds,
                origin='lower',
                extent=extent,
                cmap=cmap,
                rasterized=True
            )
    else:
        if log is True:
            im = ax.imshow(
                mmds,
                origin='lower',
                cmap=cmap,
                norm=matplotlib.colors.LogNorm(vmin=mmds.min(),
                                               vmax=mmds.max()),
                rasterized=True
            )
        else:
            im = ax.imshow(
                mmds,
                origin='lower',
                extent=extent,
                cmap=cmap,
                rasterized=True
            )
    if colorbar:
        # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.81, 0.135, 0.01, 0.715])
        fig.colorbar(im, cax=cbar_ax)

    if ref is not None:
        ax.plot(ref[0], ref[1], '*', color='white',
                markersize=20,
                markeredgecolor='black')
        # ax.set_title(f'MMD heatmap w.r.t reference point {ref}')

    if contour is not None:
        if extent is not None:
            mask = np.ma.masked_where(contour == True, mmds)
            ax.imshow(mask, extent=extent, origin='lower',
                      cmap=matplotlib.colors.LinearSegmentedColormap
                      .from_list("", ["red", "red"]),
                      rasterized=True)
        else:
            mask = np.ma.masked_where(contour == True, mmds)
            ax.imshow(mask, origin='lower',
                      cmap=matplotlib.colors.LinearSegmentedColormap
                      .from_list("", ["red", "red"]),
                      rasterized=True)
        legend_elements = [Patch(facecolor='red', edgecolor='red',
                                 label=r'$\mathrm{MMD}_b < \kappa$')]
        if legend:
            ax.legend(handles=legend_elements, loc='upper left')

    if add_traj is not None:
        ax.scatter(add_traj[:, :, 0], add_traj[:, :, 1], c='orange', s=1,
                   alpha=alpha, rasterized=True)

    if add_vect is not None:
        ax.arrow(ref[0], ref[1], add_vect[0], add_vect[1],
                 length_includes_head=True,
                 facecolor='white', edgecolor='black',
                 width=0.05)

    ax.set_xlabel(r'$x_{b,1}$')
    if legend:
        ax.set_ylabel(r'$x_{b,2}$')


if __name__ == '__main__':
    ROOT = Path(__file__).parent.parent
    RESULTS = ROOT / 'Results' / 'Duffing'
    GRAMIANS = True  # compute empirical observability Gramian at ref point

    seed_np = 0
    seed_rng = 42
    set_seeds(seed_np, seed_rng)

    # Plot all heatmaps together:
    # https://stackoverflow.com/questions/13784201/how-to-have-one-colorbar-for-all-subplots
    exps = ['Good1_105_T1_dt001_proc0_meas05_sigma1500',
            'Good1_0208_T1_dt001_proc005_meas05_sigma1500',
            'Good1_0208_T1_dt001_proc05_meas05_sigma1500']
    refs = np.array([[1, 0.5], [0.2, 0.8], [0.2, 0.8]])
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(18, 5))

    for i in range(len(exps)):
        exp = exps[i]
        ref = refs[i]  # reference point
        ax = axes.flat[i]
        if i == 0:
            legend = True
        else:
            legend = False
        if i == len(exps) - 1:
            colorbar = True
        else:
            colorbar = False

        # Logging
        experiment_folder = RESULTS / exp
        print(f'Saving results to {experiment_folder}')

        # Parameters
        dim = 2
        dim_obs = 1
        dt = 0.001
        alpha = -1
        beta = 1
        delta = 0
        init_var = 1e-2
        noise_std = 1e-1
        meas_noise_var = 5e-1
        T = 1
        N_traj_ref = 50  # nb of traj for reference
        N_traj_other = 50  # nb of traj for each grid point
        N_grid = 100
        grid_space = np.linspace(-2, 2, N_grid)
        grid = np.dstack(np.meshgrid(
            grid_space, grid_space, indexing='xy')).reshape(-1, dim)  # grid
        N_others = len(grid)
        test_alpha = 0.05
        sigma = 1500

        # Create system
        measurement = DuffingMeasurement(alpha=alpha, beta=beta)
        ref_initializer = ConstantInitializer(state=ref)
        # noise = LinearBrownianMotionNoise(sigma=noise_std * np.eye(dim))
        noise = NoNoise()  # TODO
        meas_noise = GaussianNoise(
            mean=np.zeros(dim_obs),
            var=meas_noise_var * np.eye(dim_obs)
        )
        # controller = SinusoidalController(dim=1, amplitude=0.5, pulse=1.2)
        controller = NoController()
        syst = Duffing(
            alpha=alpha,
            beta=beta,
            delta=delta,
            state_initializer=ref_initializer,
            controller=controller,
            noise=noise,
            meas=measurement,
            meas_noise=meas_noise
        )

        rejected_file = experiment_folder / 'rejected.npy'
        thresholds_file = experiment_folder / 'thresholds.npy'
        mmds_file = experiment_folder / 'mmds.npy'
        others_file = experiment_folder / 'others.npy'
        allocation_file = experiment_folder / 'test_numbers_allocation.npy'
        time_file = experiment_folder / 'time.npy'

        if rejected_file.exists():
            rejected = np.load(rejected_file)
        else:
            rejected = np.logical_not(
                np.load(experiment_folder / 'accepted.npy'))
        thresholds = np.load(thresholds_file)
        mmds = np.load(mmds_file)
        grid = np.load(others_file)
        allocation = np.load(allocation_file).astype(int)
        t = np.load(time_file)

        N_grid = int(np.round(np.sqrt(grid.shape[0])))
        dim = grid.shape[-1]
        m = grid.min()
        M = grid.max()
        extent = (m, M, m, M)

        # New plot with trajectory from reference point on top
        traj, _ = syst.get_trajectories(N_traj=1, T=60, dt=dt)

        if GRAMIANS:
            # Compute empiricial observability Gramian at ref point
            # For deterministic system only
            syst.noise = NoNoise()
            syst.meas_noise = NoNoise()

            # Generate dataset of y+i - y-i
            epsilon = 0.1
            time = np.arange(start=0., stop=T, step=dt)
            init_grid = np.concatenate((ref + epsilon * np.eye(dim),
                                        ref - epsilon * np.eye(dim)))
            meas_grid = np.zeros((init_grid.shape[0], len(time), dim_obs),
                                 dtype=float)
            for n_pt, pt in enumerate(init_grid):
                trajs_initial_state, meas_initial_state, t = \
                    generate_trajs_from_initial_state(
                        system=syst, T=T, dt=dt, initial_state=pt, N_traj=1)
                meas_grid[n_pt] = np.squeeze(meas_initial_state, 0)

            # Compute cumulative Gramian
            phi = meas_grid[:dim] - meas_grid[dim:]
            gramian = np.sum(phi.transpose((1, 0, 2)) @ phi.transpose((1, 2, 0)),
                             axis=0) / (4 * epsilon ** 2)

            # Take eigenvector of smallest eigenvalue as inobservable direction
            eigvals, eigvects = scipy.linalg.eig(gramian)
            idx_min = np.argmin(np.abs(np.real(eigvals)))
            inobs_vect = eigvects[:, idx_min]
            spec_file = experiment_folder / 'Specifications.txt'
            with spec_file.open('a') as f:
                print(f'Empirical observability Gramian: {gramian}', file=f)
                print(f'Its eigenvalues and eigenvectors: {eigvals, eigvects}',
                      file=f)

            # Plot mmd map with this vector
            fig_file = experiment_folder / \
                       'mmds_with_traj_nonoise_contour_gramian.pdf'
            plot_many_mmd(ax, fig_file=fig_file, mmds=mmds, ref=ref,
                          extent=extent, contour=rejected, add_traj=traj,
                          add_vect=inobs_vect,  alpha=0.2, colorbar=colorbar,
                          legend=legend)
            titles = [r'$(a) \; x_a = (1, 0.5)$',
                      r'$(b) \; x_a = (0.2, 0.8)$',
                      r'$(c) \; x_a = (0.2, 0.8), \, \mathrm{more} \,'
                      r'\mathrm{noise}$']
            ax.set_title(titles[i])

    plt.savefig(str(fig_file), bbox_inches="tight")
    plt.clf()
    plt.close('all')