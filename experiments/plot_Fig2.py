from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import dynamical_systems.dynamical_systems as ds
from experiments.run import compute_inobservable_direction, compute_nominal_trajectory, DuffingMeasurement

plt.rcParams['pgf.texsystem'] = 'pdflatex'
plt.rcParams['font.size'] = 16


ROOT = Path(__file__).parent.parent


def create_title(subfig_label, x_a, more_noise):
    return (
        r'(' + subfig_label + r') ' +
        r'$x_\mathrm{a}=(' + f'{x_a[0]:.1f}, {x_a[1]:.1f}' + r')$' + 
        (
            r', more noise' if more_noise else r''
        )
        )

def plot_fig2(path_a, path_b, path_c, forced_sigma):
    paths = [path_a, path_b, path_c]
    rejecteds = [None] * 3
    mmds = [None] * 3
    thresholds = [None] * 3
    grids = [None] * 3
    N_grids = [None] * 3
    sigmas = [None] * 3
    for i,path in enumerate(paths):
        if forced_sigma is not None:
            sigmas[i] = forced_sigma
        else:
            with (path / 'sigmas.npy').open('rb') as f:
                sigmas_load = np.load(f, allow_pickle=False)
                sigmas[i] = np.quantile(sigmas_load, 0.1)
        with (path / 'others.npy').open('rb') as f:
            grids[i] = np.load(f, allow_pickle=False)
        N_grids[i] = int(np.sqrt(grids[i].shape[0]))
        if N_grids[i] ** 2 != grids[i].shape[0]:
            raise ValueError(f'Grid is not square: got {np.prod(grids.shape)} elements, expected {N_grids[i]**2}')
        with (path / 'rejected.npy').open('rb') as f:
            rejecteds[i] = np.load(f, allow_pickle=False).reshape((N_grids[i], N_grids[i]))
        with (path / 'mmds.npy').open('rb') as f:
            mmds[i] = np.load(f, allow_pickle=False).reshape((N_grids[i], N_grids[i]))
        with (path / 'thresholds.npy').open('rb') as f:
            thresholds[i] = np.load(f, allow_pickle=False).reshape((N_grids[i], N_grids[i]))
        
    initial_states = [np.array([1, 0.5])] + ([np.array([0.2, 0.8])] * 2)
    process_noises = [0.05, 0.05, 0.5]
    meas_noises = [0.05, 0.05, 0.5]
    dt = 0.01
    T = 30
    system = ds.Duffing(
        alpha=-1, 
        beta=1, 
        delta=0, 
        controller=ds.NoController(), 
        meas=DuffingMeasurement(alpha=-1, beta=1), 
        state_initializer=ds.ConstantInitializer(initial_states[0]),
        noise=ds.NoNoise(),
        meas_noise=ds.NoNoise()
    )

    inobservable_directions = [None] * 3
    nominal_trajectories = [None] * 3
    for i in range(len(paths)):
        system.noise = ds.LinearBrownianMotionNoise(process_noises[i]*np.eye(2))
        system.meas_noise = ds.GaussianNoise(0, meas_noises[i])
        inobservable_directions[i] = compute_inobservable_direction(system, initial_states[i], T, dt)
        nominal_trajectories[i] = compute_nominal_trajectory(system, initial_states[i], T, dt)

    vmin = min(list(mmd.min() for mmd in mmds))
    vmax = max(list(mmd.max() for mmd in mmds))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    cmap = sns.color_palette("crest_r", as_cmap=True)
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    first_plot = True
    more_noise = [False, False, True]
    subfig_labels = 'abc'
    for i, ax in enumerate(axes.flat):
        xinf = grids[i][0,0]
        xsup = grids[i][-1,-1]
        extent = (xinf, xsup, xinf, xsup)
        ax.imshow(
            mmds[i],
            origin='lower',
            extent=extent,
            cmap=cmap,
            norm=norm,
            rasterized=True
        )
        ax.plot(
            initial_states[i][0],
            initial_states[i][1],
            '*',
            color='white',
            markersize=20,
            markeredgecolor='black'
        )
        mask = np.ma.masked_where(rejecteds[i] == True, mmds[i])
        ax.imshow(
            mask,
            extent=extent,
            origin='lower',
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["red", "red"]),
            rasterized=True
        )
        if first_plot:
            legend_elements = [Patch(facecolor='red', edgecolor='red',
                                label=r'$\widehat{\mathrm{MMD}} < \kappa$')]
            ax.legend(handles=legend_elements, loc='upper left')
            first_plot = False
            ax.set_ylabel(r'$x_{\mathrm{b},2}$')

        ax.scatter(
            nominal_trajectories[i][:, :, 0],
            nominal_trajectories[i][:, :, 1], 
            c='orange', 
            s=1,
            alpha=1, 
            rasterized=True
        )
        ax.arrow(
            initial_states[i][0], 
            initial_states[i][1], 
            inobservable_directions[i][0], 
            inobservable_directions[i][1],
            length_includes_head=True,
            facecolor='white',
            edgecolor='black',
            width=0.05
        )
        ax.set_xlabel(r'$x_{\mathrm{b},1}$')
        ax.set_title(create_title(subfig_labels[i], initial_states[i], more_noise[i]))
    fig.subplots_adjust(left=0.03, right=0.95, top=0.9, bottom=0.1, wspace=0.2)
    color_ax = fig.add_axes([0.95, 0.1, 0.015, 0.8])
    # divider = make_axes_locatable(ax)
    # color_ax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(
        mappable=cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=color_ax
    )
    # fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    fig_path = ROOT / 'Results' / 'Article' / 'Fig2.pdf'
    plt.savefig(fig_path, dpi=300, backend='pgf')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--a',
        help='Path to Figure 2a folder (relative from project root)',
        default='Results/Article/Fig2a'
    )
    parser.add_argument(
        '--b',
        help='Path to Figure 2b folder (relative from project root)',
        default='Results/Article/Fig2b'
    )
    parser.add_argument(
        '--c',
        help='Path to Figure 2c folder (relative from project root)',
        default='Results/Article/Fig2c'
    )
    parser.add_argument(
        '--sigma',
        help='Value of \sigma to use. Heuristic is used if left unspecified.',
        default=None,
        type=float
    )
    args = parser.parse_args()

    paths = [
        ROOT / path for path in [args.a, args.b, args.c]
    ]
    plot_fig2(*paths, args.sigma)