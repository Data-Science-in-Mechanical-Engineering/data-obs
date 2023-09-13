# TODO finish implementing utils and implement scripts

from pathlib import Path
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import scipy.linalg as la
from joblib import Parallel, delayed
from argparse import ArgumentParser
import numbers

import dynamical_systems.dynamical_systems as ds
from utils import JOB_SIZE, get_meas_from_to, get_initial_state_meas, log_test_results, get_new_experiment_folder, dump_specs, log_mmds, log_data, log_initial_state_data, plot_mmd
from kernels_dynamical_systems.custom_kernels import TrajectoryRBFTwoSampleTest


class DuffingMeasurement(ds.Measurement):
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
    

def generate_trajectories_from_initial_state(system, T, dt, initial_state, N_traj, *args, return_t=True, **kwargs):
    prev_initializer = system.state_initializer
    initializer = ds.ConstantInitializer(initial_state)
    system.state_initializer = initializer
    trajs, t = system.get_trajectories(N_traj, T, dt, *args, **kwargs)
    meas = system.get_output_trajectories(N_traj, T, traj=trajs)
    system.state_initializer = prev_initializer
    if return_t:
        return trajs, meas, t
    else:
        return trajs, meas


def preprocess_data_for_test(ref_meas, meas):
    scaler = StandardScaler()
    dim_obs = meas.shape[-1]
    N_traj_initial_state = ref_meas.shape[0]
    N_traj_other = meas.shape[0]
    whole_meas = np.transpose(np.concatenate(
        (ref_meas, meas), axis=1), (1, 0, 2)).reshape(-1, dim_obs)
    scaler = scaler.fit(whole_meas)
    scaled_ref_meas = scaler.transform(
        ref_meas.reshape(-1, dim_obs)).reshape(N_traj_initial_state, -1, dim_obs)
    scaled_meas = scaler.transform(
        meas.reshape(-1, dim_obs)).reshape(N_traj_other, -1, dim_obs)
    return scaled_ref_meas, scaled_meas


def compute_sigma(ref_meas, meas):
    scaled_ref_meas, scaled_meas = preprocess_data_for_test(ref_meas, meas)
    # The test computes the correct sigma at initialization
    test = TrajectoryRBFTwoSampleTest(scaled_ref_meas, scaled_meas, alpha=0.1, sigma=None)
    return test.sigma


def generate_data(experiment_folder, system, initial_state, grid, N_traj_initial_state, N_traj_others, T, dt, n_jobs):
    ## Generate trajectory of initial state
    ref_trajs, ref_meas, t = generate_trajectories_from_initial_state(
        system, T, dt, initial_state, N_traj_initial_state
    )
    log_initial_state_data(experiment_folder, ref_trajs, ref_meas)

    ## Generate trajectories of others
    inputs = [
        {
            'system':system,
            'T':T,
            'dt':dt,
            'initial_state':other,
            'N_traj':N_traj_others,
            'return_t': False
        }
        for other in grid
    ]
    sigmas = np.zeros(grid.shape[0])

    for job_start in range(0, grid.shape[0], JOB_SIZE):
        job_end = job_start + JOB_SIZE
        out = Parallel(n_jobs=n_jobs)(
            delayed(generate_trajectories_from_initial_state)(**kwargs) for kwargs in inputs[job_start:job_end]
        )
        trajs = np.array([out_item[0] for out_item in out])
        meas = np.array([out_item[1] for out_item in out])

        # save data and compute sigma
        for j in range(trajs.shape[0]):
            n_other = job_start + j
            log_data(experiment_folder, n_other, trajs[j, ...], meas[j, ...])
            sigmas[n_other] = compute_sigma(ref_meas, meas[j, ...])
    
    sigmas_file = experiment_folder / 'sigmas.npy'
    with sigmas_file.open('wb') as f:
        np.save(file=f, arr=sigmas, allow_pickle=False)

    return t, sigmas



def perform_test(ref_meas, other_meas, alpha, sigma):
    scaled_ref_meas, scaled_other_meas = preprocess_data_for_test(ref_meas, other_meas)
    test = TrajectoryRBFTwoSampleTest(
        # meas_initial_state, meas_other,
        scaled_ref_meas, scaled_other_meas,
        alpha=alpha,
        sigma=sigma
    )
    t0 = time.time()
    test.perform_test()
    t1 = time.time()
    rejected = not test.is_null_hypothesis_accepted()
    mmd = test.test_stat
    threshold = test.threshold
    return rejected, mmd, threshold, test, t1 - t0


def get_mmd_map(experiment_folder, alpha, sigma, grid, n_jobs):
    scalar_sigma = isinstance(sigma, numbers.Number)
    rejecteds = np.zeros(grid.shape[0], dtype=bool)
    mmds = np.zeros(grid.shape[0], dtype=float)
    thresholds = np.zeros(grid.shape[0], dtype=float)
    test_numbers_allocation = np.zeros(grid.shape[0], dtype=int)
    test_times = np.zeros(grid.shape[0], dtype=float)

    ref_meas = get_initial_state_meas(experiment_folder)
    for job_start in range(0, grid.shape[0], JOB_SIZE):
        job_end = job_start + JOB_SIZE
        effective_job_end = rejecteds[:job_end].shape[0]
        
        meas = get_meas_from_to(experiment_folder, job_start, job_end)
        if scalar_sigma:
            kwargs_list = [{
                'ref_meas': ref_meas,
                'other_meas': other_meas,
                'alpha': alpha,
                'sigma': sigma
            } for other_meas in meas]
        else:
            kwargs_list = [{
                'ref_meas': ref_meas,
                'other_meas': other_meas,
                'alpha': alpha,
                'sigma': None
            } for i,other_meas in enumerate(meas)]
        out = Parallel(n_jobs=n_jobs)(
            delayed(perform_test)(**kwargs) for kwargs in kwargs_list
        )
        rejecteds[job_start:job_end] = np.array([out_item[0] for out_item in out])
        mmds[job_start:job_end] = np.array([out_item[1] for out_item in out])
        thresholds[job_start:job_end] = np.array([out_item[2] for out_item in out])
        test_numbers_allocation[job_start:job_end] = np.array(range(job_start, effective_job_end))
        test_times[job_start:job_end] = np.array([out_item[4] for out_item in out])

        for j, out_item in enumerate(out):
            _, _, _, test, test_time = out_item
            log_test_results(
                experiment_folder, 
                test_numbers_allocation[job_start+j],
                test, 
                test_time, 
            )
    return rejecteds, mmds, thresholds, test_times, test_numbers_allocation


def compute_inobservable_direction(system, ref, T, dt):
    # Compute empiricial observability Gramian at ref point
    # For deterministic system only
    system.noise = ds.NoNoise()
    system.meas_noise = ds.NoNoise()
    dim = system.dim
    dim_obs = 1

    # Generate dataset of y+i - y-i
    epsilon = 0.1
    time = np.arange(start=0., stop=T, step=dt)
    init_grid = np.concatenate((ref + epsilon * np.eye(dim),
                                ref - epsilon * np.eye(dim)))
    meas_grid = np.zeros((init_grid.shape[0], len(time), dim_obs),
                            dtype=float)
    for n_pt, pt in enumerate(init_grid):
        _, meas_initial_state = \
            generate_trajectories_from_initial_state(
                system=system, T=T, dt=dt, initial_state=pt, N_traj=1, return_t=False)
        meas_grid[n_pt] = np.squeeze(meas_initial_state, 0)

    # Compute cumulative Gramian
    phi = meas_grid[:dim] - meas_grid[dim:]
    gramian = np.sum(phi.transpose((1, 0, 2)) @ phi.transpose((1, 2, 0)),
                        axis=0) / (4 * epsilon ** 2)

    # Take eigenvector of smallest eigenvalue as inobservable direction
    eigvals, eigvects = la.eig(gramian)
    idx_min = np.argmin(np.abs(np.real(eigvals)))
    inobs_vect = eigvects[:, idx_min]
    return inobs_vect

def compute_nominal_trajectory(system, ref, T, dt):
    original_process_noise = system.noise
    original_meas_noise = system.meas_noise
    original_ref = system.state_initializer
    system.noise = ds.NoNoise()
    system.meas_noise = ds.NoNoise()
    system.state_initializer = ds.ConstantInitializer(ref)
    nominal_trajectory, _ = system.get_trajectories(N_traj=1, T=T, dt=dt)
    system.noise = original_process_noise
    system.mea_noise = original_meas_noise
    system.state_initializer = original_ref
    return nominal_trajectory


def run(sys_type, choice_initial_state, process_noise_var, meas_noise_var, N_grid, N_traj, N_traj_ref, T, dt, sigma, alpha, no_gramian, no_trajectory, n_jobs, plot_only):
    ROOT = Path(__file__).parent.parent
    dim = 2
    if sys_type == 'linear':
        RESULTS = ROOT / 'Results' / 'Linear'
        SystemConstructor = ds.ContinuousTimeLTI
        initial_state = np.array([1.5, 0.5])
        kwargs_system = {
            'dim': dim,
            'A': np.array([
                [-2, -1.],
                [-1., -2.]
            ]), # eigenvalues -1 and -3, eigenvectors [-1, 1] and [1, 1]
            'B': np.eye(dim),
            'controller': ds.SinusoidalController(
                dim=dim, pulse=2., phase=0., amplitude=3.
            ),
            'meas': ds.LinearMeasurement(np.array([-1., 1.])),
        }
        PLOT_GRAMIAN = False
        PLOT_TRAJECTORY = False
    elif sys_type == 'duffing':
        RESULTS = ROOT / 'Results' / 'Duffing'
        SystemConstructor = ds.Duffing
        initial_state = np.array([1, 0.5]) if choice_initial_state == 1 else np.array([0.2, 0.8])
        kwargs_system = {
            'alpha': -1,
            'beta': 1,
            'delta': 0,
            'controller': ds.NoController(),
            'meas': DuffingMeasurement(alpha=-1, beta=1)
        }
        PLOT_GRAMIAN = not no_gramian
        PLOT_TRAJECTORY = not no_trajectory
    else:
        raise ValueError(f'Invalid system type {sys_type}.')

    kwargs_system.update({
        'state_initializer': ds.ConstantInitializer(initial_state),
        'noise': ds.LinearBrownianMotionNoise(sigma=np.sqrt(process_noise_var)*np.eye(dim)),
        'meas_noise': ds.GaussianNoise(0., meas_noise_var),
    })
    system = SystemConstructor(**kwargs_system)
    
    if plot_only is None:
        experiment_folder = get_new_experiment_folder(RESULTS)
        print(f'Saving results to {experiment_folder}')
        xinf = -2
        xsup = 2
        grid_space = np.linspace(xinf, xsup, N_grid)
        grid = np.dstack(np.meshgrid(
            grid_space, grid_space, indexing='xy')).reshape(-1, dim)  # grid

        t, sigmas = generate_data(
            experiment_folder=experiment_folder,
            system=system,
            initial_state=initial_state,
            grid=grid,
            N_traj_initial_state=N_traj_ref,
            N_traj_others=N_traj,
            T=T,
            dt=dt,
            n_jobs=n_jobs,
        )
    else:
        experiment_folder = RESULTS / plot_only
        with (experiment_folder / 'sigmas.npy').open('rb') as f:
            sigmas = np.load(f, allow_pickle=False)
        with (experiment_folder / 'time.npy').open('rb') as f:
            t = np.load(f, allow_pickle=False)
        with (experiment_folder / 'others.npy').open('rb') as f:
            grid = np.load(f, allow_pickle=False)
        N_grid = int(np.round(np.sqrt(grid.shape[0]),0))
        xinf = grid[0,0]
        xsup = grid[-1,-1]


    if sigma is None:
        sigma = np.quantile(sigmas, 0.1)
    if not isinstance(sigmas, np.ndarray) and sigmas == 'individual':
        # Instance check is necessary to silence warning when comparing numpy array and string
        sigma = sigmas
        
    if plot_only is None:
        dump_specs(experiment_folder, {k:v for k,v in locals().items() if k not in ['t','sigmas']})  # logging


    rejecteds, mmds, thresholds, _, test_numbers_allocation = get_mmd_map(
        experiment_folder=experiment_folder,
        alpha=alpha,
        sigma=sigma,
        grid=grid,
        n_jobs=n_jobs
    )
    rejecteds = rejecteds.reshape((N_grid, N_grid))
    mmds = mmds.reshape((N_grid, N_grid))
    thresholds = thresholds.reshape((N_grid, N_grid))
    sigmas = sigmas.reshape((N_grid, N_grid))


    if PLOT_GRAMIAN:
        inobservable_direction = compute_inobservable_direction(system, initial_state, T, dt)
    else:
        inobservable_direction = None
    if PLOT_TRAJECTORY:
        nominal_trajectory = compute_nominal_trajectory(system, initial_state, 60, dt)
    else:
        nominal_trajectory = None

    log_mmds(
        experiment_folder, 
        t, 
        mmds, 
        rejecteds, 
        thresholds, 
        test_numbers_allocation, 
        grid, 
        extent=(xinf, xsup, xinf, xsup),
        ref=initial_state,
        add_vect=inobservable_direction
    )

    plot_mmd(
        fig_file=experiment_folder / 'mmds.pdf', 
        mmds=mmds, 
        ref=initial_state, 
        extent=(xinf, xsup, xinf, xsup), 
        add_vect=inobservable_direction,
        add_traj=nominal_trajectory
    )
    plot_mmd(
        fig_file=experiment_folder / 'mmds_with_rejected.pdf', 
        mmds=mmds, 
        ref=initial_state, 
        extent=(xinf, xsup, xinf, xsup), 
        add_vect=inobservable_direction,
        contour=rejecteds,
        add_traj=nominal_trajectory
    )
    plot_mmd(
        fig_file=experiment_folder / 'thresholds.pdf', 
        mmds=thresholds, 
        ref=initial_state, 
        extent=(xinf, xsup, xinf, xsup),
    )
    if plot_only is None:
        plot_mmd(
            fig_file=experiment_folder / 'sigmas.pdf', 
            mmds=sigmas, 
            ref=initial_state, 
            extent=(xinf, xsup, xinf, xsup),
        )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--system",
        help="Either 'linear' or 'duffing'",
        type=str
    )
    parser.add_argument(
        "--initial_state",
        help="Either 1 or 2; choice of initial state. Ignored if --system is linear. If --system is duffing, 1 corresponds to x_a = (1, 0.5) and 2 to x_a = (0.2, 0.8).",
        type=int
    )
    parser.add_argument(
        "--process_noise",
        help="Process noise variance. Only scalar variance is supported.",
        type=float
    )
    parser.add_argument(
        "--measurement_noise",
        help="Measurement noise variance. Only scalar variance is supported.",
        type=float
    )
    parser.add_argument(
        "--n_grid",
        help='Size of the side of the grid.',
        type=int
    )
    parser.add_argument(
        '--n_traj',
        help='Number of independent trajectories from each initial state.',
        type=int
    )
    parser.add_argument(
        '--n_traj_ref',
        help='Number of independent trajectories from the initial state. Defaults to n_traj_others if unspecified.',
        type=int
    )
    parser.add_argument(
        '--T',
        help='Simulation time, in seconds.',
        type=float
    )
    parser.add_argument(
        '--dt',
        help='Simulation step size, in seconds.',
        type=float
    )
    parser.add_argument(
        '--sigma',
        help="Value of \sigma to pick. Defaults to using the heuristic if unspecified. Can also be set to `individual` to use an individual value of \sigma for each pair of initial states.",
        default=None,
    )
    parser.add_argument(
        '--alpha',
        help='Level of the test',
        type=float
    )
    parser.add_argument(
        '--no_gramian',
        help='Activating this flag disables the computation and plotting of the Gramian. Only relevant when --system is set to duffing',
        action="store_true",
    )
    parser.add_argument(
        '--no_trajectory',
        help='Activating this flag disables the computation and plotting of the nominal trajectory. Only relevant when --system is set to duffing',
        action='store_true'
    )
    parser.add_argument(
        '--n_jobs',
        help='Number of jobs for multiprocessing. Defaults to 1.',
        type=int,
        default=1
    )
    parser.add_argument(
        '--seed',
        help='Random seed',
        type=int,
        default=0
    )
    parser.add_argument(
        '--plot_only',
        help='Whether to skip the simulation and instead only plot the experiment with folder name passed as parameter.',
        type=str
    )

    args = parser.parse_args()

    ds.set_seeds(args.seed, args.seed+1)

    run(
        args.system, 
        args.initial_state,
        args.process_noise, 
        args.measurement_noise, 
        args.n_grid, 
        args.n_traj, 
        args.n_traj_ref if args.n_traj_ref is not None else args.n_traj, 
        args.T, 
        args.dt, 
        args.sigma, 
        args.alpha,
        args.no_gramian,
        args.no_trajectory,
        args.n_jobs,
        args.plot_only,
    )