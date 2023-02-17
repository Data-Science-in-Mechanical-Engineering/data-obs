import numpy as np
import scipy.integrate
import warnings

from ...noise import LinearBrownianMotionNoise, GaussianNoise, NoNoise
from ...state_initializer import GaussianInitializer

from ...dynamical_system import ContinuousTimeSystem
from .continuous_observer import ContinuousTimeObserver

from ...utils import identity_or_broadcast, interpolate_trajectory


class ContinuousTimeEKF(ContinuousTimeObserver):
    def __init__(self, dim, system: ContinuousTimeSystem, Q=None, R=None, P0=None, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.system = system
        meas_dim = system.meas.dim
        if Q is None and not isinstance(system.noise, LinearBrownianMotionNoise):
            raise ValueError(
                'Cannot use ground truth value of Q for EKF since the system does '
                'not have LinearBrownianMotionNoise. Please specify Q.'
            )
        self.Q = identity_or_broadcast(
            Q, dim) if Q is not None else system.noise.sigma
        if R is None and not isinstance(system.meas_noise, GaussianNoise):
            raise ValueError(
                'Cannot use ground truth value of R for EKF since the system does not have'
                'Gaussian measurement noise. Please specify R.'
            )
        self.R = identity_or_broadcast(
            R, dim) if R is not None else system.meas_noise.var
        if P0 is None and not isinstance(system.state_initializer, GaussianInitializer):
            raise ValueError(
                'Cannot use ground truth value of P0 for EKF since the system does not have'
                'Gaussian initializations. Please specify P0.'
            )
        self.P0 = identity_or_broadcast(
            P0, dim) if P0 is not None else system.state_initializer.var

    # Compute estimations from EKF
    # Expects inputs of shape (Ntraj, T, D)
    def get_observations(self, t, measurements, controls=None, return_scipy_output=False, *args, **kwargs):
        # TODO factorize this
        observations = np.zeros(
            measurements.shape[:-1] + (self.dim,), dtype=float)
        if return_scipy_output:
            scipy_output = [None] * measurements.shape[0]

        for n, meas_traj in enumerate(measurements):
            traj_controller = interpolate_trajectory(
                t, controls[n], kind='previous')
            y = interpolate_trajectory(t, meas_traj)

            def obs_dynamics_for_traj_n(t, x_hat_P: np.ndarray):
                x_hat = x_hat_P[:self.system.dim]
                P = x_hat_P[self.dim:].reshape(
                    self.system.dim, self.system.dim)
                F = self.system.jacobian(x_hat, traj_controller(t), t)
                H = self.system.meas.jacobian(x_hat, t)
                K = P @ H.T @ np.linalg.inv(self.R)
                y_hat = self.system.meas.get_measurement(x_hat)

                x_hat_dot = self.system.f(x_hat, t) + K @ (y(t) - y_hat)
                P_dot = F @ P + P @ F.T - K @ H @ P + self.Q

                P_dot_flat = P_dot.reshape(-1)
                x_hat_dot_P_dot = np.hstack((x_hat_dot, P_dot_flat))
                return x_hat_dot_P_dot

            initial_xhat = self.state_initializer.initial_state()
            initial_P = self.P0.reshape(-1)
            initial_xhat_P = np.hstack((initial_xhat, initial_P))

            obs = scipy.integrate.solve_ivp(
                fun=obs_dynamics_for_traj_n,
                t_span=(t[0], t[-1]),
                y0=initial_xhat_P,
                t_eval=t,
                *args,
                **kwargs
            )
            obs_xhat_P = obs['y'].T  # scipy outputs shape (D, T)
            obs_traj = obs_xhat_P[:, :self.dim]
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
