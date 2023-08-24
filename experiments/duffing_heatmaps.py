import numpy as np
from pathlib import Path

import os
import sys
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
import numpy as np

sys.path.append('.')

# Dirty fix conda env probl
# https://stackoverflow.com/questions/55714135/how-can-i-fix-an-omp-error-15-initializing-libiomp5-dylib-but-found-libomp
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from dynamical_systems.dynamical_systems import set_seeds, Duffing, \
    Measurement, GaussianNoise, ConstantInitializer, LinearBrownianMotionNoise, \
    NoController, NoNoise
from drift_utils import get_new_experiment_folder, dump_specs, log_mmds, \
    plot_mmd
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


if __name__ == '__main__':
    PLOT = True
    ROOT = Path(__file__).parent.parent
    RESULTS = ROOT / 'Results' / 'Duffing'
    NEW = True  # read existing experiment
    GRAMIANS = True  # compute empirical observability Gramian at ref point
    TEST = False  # test two arbitrary indistinguishable points with many trajs

    seed_np = 0
    seed_rng = 42
    set_seeds(seed_np, seed_rng)

    # Logging
    if NEW:
        experiment_folder = get_new_experiment_folder(RESULTS)
    else:
        experiment_folder = RESULTS / 'exp_0'
    print(f'Saving results to {experiment_folder}')

    # Parameters
    dim = 2
    dim_obs = 1
    dt = 0.001
    alpha = -1
    beta = 1
    delta = 0
    ref = np.array([1, 0.5])  # reference point
    init_var = 1e-2
    noise_std = 5e-2
    meas_noise_var = 5e-1
    T = 1
    N_traj_ref = 10  # nb of traj for reference
    N_traj_other = 10  # nb of traj for each grid point
    N_grid = 10
    grid_space = np.linspace(-2, 2, N_grid)
    grid = np.dstack(np.meshgrid(
        grid_space, grid_space, indexing='xy')).reshape(-1, dim)  # grid
    N_others = len(grid)
    test_alpha = 0.05
    sigma = 1500

    # Create system
    measurement = DuffingMeasurement(alpha=alpha, beta=beta)
    ref_initializer = ConstantInitializer(state=ref)
    noise = LinearBrownianMotionNoise(sigma=noise_std * np.eye(dim))
    # noise = NoNoise()
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

    # Generate MMD heatmaps
    if NEW:
        dump_specs(experiment_folder, locals().copy())  # logging
        mmd_map_results = get_mmd_map(
            system=syst,
            initial_state=ref,
            others=grid,
            N_traj_initial_state=N_traj_ref,
            N_traj_others=N_traj_other,
            T=T,
            dt=dt,
            alpha=test_alpha,
            experiment_folder=experiment_folder,
            sigma=sigma
        )
        t = mmd_map_results['t']
        mmds = mmd_map_results['mmds'].reshape(N_grid, N_grid)
        rejected = mmd_map_results['rejected'].reshape(N_grid, N_grid)
        thresholds = mmd_map_results['thresholds'].reshape(N_grid, N_grid)
        test_numbers_allocation = mmd_map_results['test_numbers_allocation']
        if sigma is None:
            # Save map of sigma values
            sigmas = mmd_map_results['sigmas'].reshape(N_grid, N_grid)
            sigmas_file = experiment_folder / 'sigmas.npy'
            with sigmas_file.open('wb') as f:
                np.save(file=f, arr=sigmas, allow_pickle=False)
        xinf = grid_space[0]
        xsup = grid_space[-1]
        extent = (xinf, xsup, xinf, xsup)
        log_mmds(experiment_folder, t, mmds, rejected, thresholds,
                 test_numbers_allocation, grid, extent=extent, ref=ref)
    else:
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

        log_mmds(experiment_folder, t, mmds, rejected, thresholds,
                 allocation, grid, extent=extent, ref=ref)

    # New plot with trajectory from reference point on top
    fig_file = experiment_folder / 'mmds_with_traj.pdf'
    traj, _ = syst.get_trajectories(N_traj=1, T=60, dt=dt)
    plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
             add_traj=traj)
    # Same without noise
    fig_file = experiment_folder / 'mmds_with_traj_nonoise.pdf'
    syst.noise = NoNoise()
    syst.meas_noise = NoNoise()
    traj, _ = syst.get_trajectories(N_traj=1, T=30, dt=dt)
    plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
             add_traj=traj)
    # Add contour of test threshold on top of plot
    fig_file = experiment_folder / 'mmds_with_contour.pdf'
    plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
             contour=rejected)
    # Add contour of test threshold on top of plot and noiseless trajectory
    fig_file = experiment_folder / 'mmds_with_traj_nonoise_contour.pdf'
    plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
             contour=rejected, add_traj=traj, alpha=0.2)
    # Add data starting at reference point on top of plot
    fig_file = experiment_folder / 'mmds_with_traj_contour_refdata.pdf'
    reftraj = np.load(os.path.join(experiment_folder,
                                   'data',  'nb0_INIT_trajs.npy'))
    plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
             contour=rejected, add_traj=reftraj, alpha=0.2)

    # Compute quantiles of sigmas
    if sigma is None:
        sigmas = np.load(experiment_folder / 'sigmas.npy')
        spec_file = experiment_folder / 'Specifications.txt'
        with spec_file.open('a') as f:
            print(f'Sigmas quantiles (0.1, 0.5, 0.9): '
                  f'{np.quantile(sigmas, (0.1, 0.5, 0.9))}', file=f)
        sigmas_fig = experiment_folder / 'sigmas.pdf'
        plot_mmd(fig_file=sigmas_fig, mmds=sigmas, ref=ref, extent=extent)

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
        plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent,
                 contour=rejected, add_traj=traj, add_vect=inobs_vect,
                 alpha=0.2)

    # Test distinguishability of two indistinguishable points with many trajs
    if TEST:
        x1 = np.array([0, 1])
        x2 = np.array([np.sqrt(1+np.sqrt(3)), 0])
        Ntraj1 = 1000
        Ntraj2 = 1000
        sigma = None
        syst.noise = NoNoise()
        syst.meas_noise = GaussianNoise(
            mean=np.zeros(dim_obs),
            var=meas_noise_var * np.eye(dim_obs)
        )
        # syst.meas_noise = NoNoise()
        traj1, meas1, t1 = \
            generate_trajs_from_initial_state(
                system=syst, T=T, dt=dt, initial_state=x1, N_traj=Ntraj1)
        traj2, meas2, t2 = \
            generate_trajs_from_initial_state(
                system=syst, T=T, dt=dt, initial_state=x2, N_traj=Ntraj1)
        from sklearn.preprocessing import StandardScaler
        from kernels_dynamical_systems.custom_kernels import \
            TrajectoryRBFTwoSampleTest
        from test_utils import print_test_results
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
                                          sigma=sigma)
        test.perform_test(verbose=True)
        result = test.is_null_hypothesis_accepted()
        spec_file = experiment_folder / 'Specifications.txt'
        correct = True if result == True else False
        print_test_results(file=spec_file, nb=1, test=test,
                           correct=correct, data_time=0, test_time=0)