import numpy as np
from ...dynamical_system import ContinuousTimeSystem


class ContinuousTimeLTI(ContinuousTimeSystem):
    def __init__(self, dim, A, B, state_initializer, controller=None, noise=None, meas=None, meas_noise=None):
        super().__init__(
            dim=dim,
            state_initializer=state_initializer,
            controller=controller,
            noise=noise,
            meas=meas,
            meas_noise=meas_noise
        )
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)

        dim_match = (self.A.shape == (dim, dim)) and (
            self.B.shape[0] == dim)
        if not dim_match:
            raise ValueError("Dimension mismatch")

    def f(self, state, t):
        u = self.controller.get_control_input(state, t)
        f_val = self.A @ state + self.B @ u
        return f_val

    def jacobian(self, state, control, t):
        return self.A
