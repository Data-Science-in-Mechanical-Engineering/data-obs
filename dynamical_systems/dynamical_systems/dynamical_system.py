import numpy as np
import sdeint
import scipy.integrate
from collections.abc import Iterable

from .noise import BrownianMotionNoise, NoNoise
from .controller import NoController
from .measurement import FullStateMeasurement
from .utils import RNG

class DynamicalSystem:
    def __init__(self, dim, state_initializer, controller=None, noise=None, meas=None, meas_noise=None, observers=None):
        self.dim = dim
        self.__state_initializer = state_initializer
        self.__controller = controller if controller is not None else NoController()
        self.__noise = noise if noise is not None else NoNoise()
        self.__meas = meas if meas is not None else FullStateMeasurement(
            dim=dim)
        self.__meas_noise = meas_noise if meas_noise is not None else NoNoise()

    @property
    def has_measurement_noise(self):
        return not isinstance(self.meas_noise, NoNoise)

    @property
    def has_process_noise(self):
        return not isinstance(self.noise, NoNoise)

    @property
    def controller(self):
        return self.__controller

    @controller.setter
    def controller(self, controller):
        self.__controller = controller

    @property
    def state_initializer(self):
        return self.__state_initializer

    @state_initializer.setter
    def state_initializer(self, state_initializer):
        self.__state_initializer = state_initializer

    @property
    def noise(self):
        return self.__noise

    @noise.setter
    def noise(self, noise):
        self.__noise = noise

    @property
    def meas(self):
        return self.__meas

    @meas.setter
    def meas(self, meas):
        self.__meas = meas

    @property
    def meas_noise(self):
        return self.__meas_noise

    @meas_noise.setter
    def meas_noise(self, meas_noise):
        self.__meas_noise = meas_noise

    def get_trajectories(self, N_traj, T, return_controls=False, *args, **kwargs):
        raise NotImplementedError

    def get_output_trajectories(self, N_traj, T, traj=None, *args, **kwargs):
        if traj is None:
            traj = self.get_trajectories(
                N_traj, T, return_controls=True, *args, **kwargs)[0]
        output_traj = self.meas.get_measurement(
            traj)  # TODO if meas depends u?
        # TODO put measurement noise directly in measurement to handle non-additive noise
        output_traj += self.meas_noise.get_noise_input(output_traj)
        return output_traj


class DiscreteTimeSystem(DynamicalSystem):
    def get_next_state_control(self, state, *args, **kwargs):
        raise NotImplementedError

    def get_trajectories(self, N_traj, T, return_controls=False, *args, **kwargs):
        traj = np.zeros((N_traj, T, self.dim))
        controls = np.zeros((N_traj, T, self.controller.dim))
        for n_traj in range(N_traj):
            traj[n_traj, 0, :] = self.state_initializer.initial_state()
            for t in range(T-1):
                traj[n_traj, t + 1], controls[n_traj, t] = \
                    self.get_next_state_control(
                        traj[n_traj, t], t, *args, **kwargs)
        if return_controls:
            return traj, controls
        return traj


class ContinuousTimeSystem(DynamicalSystem):
    def f(self, state, t):
        raise NotImplementedError

    def G(self, state, t):
        if self.has_process_noise:
            if not isinstance(self.noise, BrownianMotionNoise):
                raise ValueError(
                    'The process noise of a ContinuousTimeSystem should be a BrownianMotionNoise'
                )
            return self.noise.G(state, t)
        else:
            return 0

    def jacobian(self, state, control, t):
        raise NotImplementedError

    def get_trajectories(self, N_traj, T, dt=None, method=None, return_scipy_output=False, return_controls=False, **solver_kwargs):
        if self.has_process_noise and dt is None:
            raise ValueError('Cannot have dt=None with process noise')
        if return_scipy_output:
            if self.has_process_noise:
                raise ValueError('Cannot return scipy output for SDEs')
            scipy_output = []

        if isinstance(T, Iterable):
            T0, T = T
        else:
            T0 = 0

        initial_states = [self.state_initializer.initial_state()
                          for n in range(N_traj)]

        t_span = np.arange(
            start=T0, stop=T, step=dt) if dt is not None else None
        trajs = [None] * N_traj

        if self.has_process_noise:
            ### ORIGINAL
            for n in range(N_traj):
                trajectory = sdeint.itoint(
                    f=self.f,
                    G=self.G,
                    y0=initial_states[n],
                    tspan=t_span
                )
                trajs[n] = trajectory

            ### VECTORIZED
            # def simulate(initial_state):
            #     return sdeint.itoint(
            #         f=self.f,
            #         G=self.G,
            #         y0=initial_state,
            #         tspan=t_span
            #     )
            # simulate_vect = np.vectorize(simulate)
            # trajs = simulate_vect(np.array(initial_states))

            ### MULTIPROCESSING
            # inputs = [
            #     [
            #         'f':self.f,
            #         'G':self.G,
            #         'y0':initial_states[n],
            #         'tspan':t_span,
            #         # give a different RNG to each process
            #         'generator':np.random.default_rng(seed=RNG.get().integers(1e6))
            #     ]
            #     for n in range(N_traj)
            # ]
            # maximum_available_processes = 4
            # with multiprocessing.Pool(processes=maximum_available_processes) as pool:
            #     trajs = pool.starmap(sdeint.itoint, inputs)

            ### JOBLIB
            # inputs = [
            #     {
            #         'f':self.f,
            #         'G':self.G,
            #         'y0':initial_states[n],
            #         'tspan':t_span,
            #         # give a different RNG to each process
            #         # 'generator':np.random.default_rng(seed=RNG.get().integers(1e6))
            #     }
            #     for n in range(N_traj)
            # ]
            # trajs = Parallel(n_jobs=6)(delayed(sdeint.itoint)(**kwargs) for kwargs in inputs)
        else:
            def dynamics(t, x):
                # Reverse arguments for compatibility with Scipy
                return self.f(x, t)

            for n in range(N_traj):
                trajectory = scipy.integrate.solve_ivp(
                    dynamics,
                    # definition interval, called t_span in scipy
                    t_span=(T0, T),
                    y0=initial_states[n],
                    method='RK45' if method is None else method,
                    t_eval=t_span,  # times at which the solution is returned
                    **solver_kwargs
                )
                trajs[n] = trajectory['y'].T  # scipy outputs shape (D, T)
                if t_span is None:
                    t_span = trajectory['t']

                if return_scipy_output:
                    scipy_output.append(trajs)

        trajs = np.array(trajs)
        output = (trajs, t_span)
        if return_controls:
            controls = self.controller.get_control_input(trajs, t_span)
            output += (controls,)
        if return_scipy_output:
            output += (scipy_output,)
        return output
