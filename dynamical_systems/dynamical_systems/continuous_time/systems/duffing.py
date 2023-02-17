import numpy as np
from ...dynamical_system import ContinuousTimeSystem


class Duffing(ContinuousTimeSystem):
    def __init__(self, alpha, beta, delta, state_initializer, controller=None, noise=None, meas=None, meas_noise=None):
        super().__init__(
            dim=2,
            state_initializer=state_initializer,
            controller=controller,
            noise=noise,
            meas=meas,
            meas_noise=meas_noise
        )
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        # Hardcode solver options for paper plots
        self.solver_kwargs = {'rtol': 1e-5, 'atol': 1e-8}

    def f(self, state, t):
        u = self.controller.get_control_input(state, t)
        xdot = np.zeros_like(state)
        xdot[..., 0] = state[..., 1]
        xdot[..., 1] = - self.alpha * state[..., 0] - self.beta * state[..., 0] ** 3 - self.delta * state[..., 1] + u
        return xdot

    def jacobian(self, state, control, t):
        A = np.array([
            [0., 1.],
            [- self.alpha - 3 * self.beta * state[..., 0] ** 2, - self.delta]
        ])
        return A

    def get_trajectories(self, N_traj, T, dt=None, method=None,
                         return_scipy_output=False, return_controls=False,
                         **solver_kwargs):
        return super().get_trajectories(
            N_traj=N_traj, T=T, dt=dt, method=method,
            return_scipy_output=return_scipy_output,
            return_controls=return_controls, **self.solver_kwargs)
