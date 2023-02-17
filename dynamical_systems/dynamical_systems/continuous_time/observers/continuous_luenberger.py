import scipy.integrate
import numpy as np
import warnings

from ...dynamical_system import ContinuousTimeSystem
from .continuous_observer import ContinuousTimeObserver
from ...utils import interpolate_trajectory


class ContinuousTimeLuenberger(ContinuousTimeObserver):
    def __init__(self, dim: int, L: np.ndarray, system: ContinuousTimeSystem, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.L = np.atleast_2d(L)
        self.system = system
        if self.L.shape != (self.system.dim, self.system.meas.dim):
            raise ValueError('Size mismatch')

    def get_observations(self, t, measurements, controls=None, return_scipy_output=False, *args, **kwargs):
        # TODO factorize this
        observations = np.zeros(
            measurements.shape[:-1] + (self.dim,), dtype=float)
        if return_scipy_output:
            scipy_output = [None] * measurements.shape[0]

        for n, meas_traj in enumerate(measurements):
            y = interpolate_trajectory(t, meas_traj)

            def obs_dynamics_for_traj_n(t, x_hat):
                y_hat = self.system.meas.get_measurement(x_hat)
                x_hat_dot = self.system.f(x_hat, t) + self.L @ (y(t) - y_hat)
                return x_hat_dot

            obs = scipy.integrate.solve_ivp(
                fun=obs_dynamics_for_traj_n,
                t_span=(t[0], t[-1]),
                y0=self.state_initializer.initial_state(),
                t_eval=t,
                *args,
                **kwargs
            )
            obs_traj = obs['y'].T  # scipy outputs shape (D, T)
            if not obs['success']:
                warnings.warn(
                    f'Observer ODE solver failed. Reason:\n{obs["message"]}.')
                solved = obs_traj
                obs_traj = np.zeros_like(observations[n])
                obs_traj[:solved.shape[0], :] = solved
                obs_traj[solved.shape[0]:, :] = np.nan

            observations[n] = obs_traj
            if return_scipy_output:
                scipy_output[n] = obs

        if return_scipy_output:
            return observations, scipy_output
        return observations
