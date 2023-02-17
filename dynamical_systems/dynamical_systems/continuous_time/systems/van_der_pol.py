import numpy as np
from ...dynamical_system import ContinuousTimeSystem


class VanDerPol(ContinuousTimeSystem):
    def __init__(self, mu, state_initializer, controller=None, noise=None, meas=None, meas_noise=None):
        super().__init__(
            dim=2,
            state_initializer=state_initializer,
            controller=controller,
            noise=noise,
            meas=meas,
            meas_noise=meas_noise
        )
        self.mu = mu

    def f(self, state, t):
        u = self.controller.get_control_input(state, t)
        f_val = np.array([
            state[1],
            self.mu * (1 - state[0]**2) * state[1] - state[0]
        ]) + u
        return f_val

    def jacobian(self, state, control, t):
        A = np.array([
            [0., 1.],
            [-2*self.mu*state[1] - 1, self.mu*(1 - state[0]**2)]
        ])
        return A
