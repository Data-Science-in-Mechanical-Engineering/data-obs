import numpy as np
from scipy.linalg import solve_discrete_are, inv


class FeedbackController:
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim

    def get_control_input(self, state, t):
        raise NotImplementedError


class NoController(FeedbackController):
    def __init__(self, dim=None, *args, **kwargs):
        if dim is None:
            dim = 1
        super().__init__(dim=dim, *args, **kwargs)

    def get_control_input(self, state, t):
        shape = state.shape[:-1] + (self.dim,)
        return np.zeros(shape)


class LinearController(FeedbackController):
    def __init__(self, K, *args, **kwargs):
        dim = K.shape[1]
        super().__init__(dim=dim, *args, **kwargs)
        self.K = K

    def get_control_input(self, state, t):
        return - self.K @ state

class SinusoidalController(FeedbackController):
    def __init__(self, dim=None, pulse=1, phase=0, amplitude=1, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        # Either dim is given, or use same dim as state
        if isinstance(pulse, np.ndarray) and not isinstance(phase, np.ndarray):
            phase = np.ones_like(pulse) * phase
        elif isinstance(phase, np.ndarray) and not isinstance(pulse, np.ndarray):
            pulse = np.ones_like(phase) * pulse
        elif isinstance(phase, np.ndarray) and isinstance(pulse, np.ndarray) and phase.shape != pulse.shape:
            raise ValueError(
                "Dimension mismatch; pulse and phase should have the same shape")

        self.pulse = pulse
        self.phase = phase
        self.amplitude = amplitude

    def get_control_input(self, state, t):
        control_value = self.amplitude * np.sin(self.pulse * t + self.phase)
        if not self.dim:
            state_broadcast = np.ones_like(state)
        else:
            if len(state.shape) == 2:
                state_broadcast = np.ones((state.shape[0], self.dim))
            else:
                state_broadcast = np.ones((self.dim,))
        # if both arrays, element-wise multiplication by 1 is harmless
        return control_value * state_broadcast


class DLQRController(LinearController):
    def __init__(self, A, B, Q, R, *args, **kwargs):
        K = DLQRController.solve_dlqr(A, B, Q, R)
        super().__init__(K, *args, **kwargs)

    @staticmethod
    def solve_dlqr(A, B, Q, R):
        """Solve the discrete time lqr controller.
        x[k+1] = A x[k] + B u[k]
        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        # ref Bertsekas, p.151
        R = np.atleast_2d(R)
        # first, try to solve the ricatti equation
        X = np.array(solve_discrete_are(A, B, Q, R))
        # compute the LQR gain
        K = np.array(inv(B.T @ X @ B + R) @ (B.T @ X @ A))
        # eigVals, eigVecs = scipy.linalg.eig(A - B @ K)
        return K
