import numpy as np
from pyparsing import line


class Measurement:
    def __init__(self, dim, *args, **kwargs):
        self.dim = dim

    def jacobian(self, state, t):
        raise NotImplementedError

    def get_measurement(self, state, *args, **kwargs):
        raise NotImplementedError


class FullStateMeasurement(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def jacobian(self, state, t):
        return np.eye(self.dim)

    def get_measurement(self, state, *args, **kwargs):
        return state


class LinearMeasurement(Measurement):
    def __init__(self, C, D=None, *args, **kwargs):
        dim = C.shape[0]
        super().__init__(dim=dim, *args, **kwargs)
        self.C = C
        self.D = D

        if (D is not None) and not (C.shape[0] == D.shape[0]):
            raise ValueError("Dimension mismatch")
        if len(self.C.shape) < 2:  # C of shape (dim_y, dim_x)
            self.C = self.C.reshape(1, -1)
        if self.D is not None:
            if len(self.D.shape) < 2:  # D of shape (dim_y, dim_u)
                self.D = self.D.reshape(1, -1)

    def jacobian(self, state, t):
        return self.C

    def get_measurement(self, state, *args, **kwargs):
        if self.D is None:
            return state @ self.C.T
        else:
            u = args[0]
            return np.expand_dims(state @ self.C.T + u @ self.D.T, -1)


class QuadraticFormMeasurement(Measurement):
    def __init__(self, Q, S=None, *args, **kwargs):
        super().__init__(dim=1, *args, **kwargs)
        self.Q = Q
        self.S = S if S is not None else np.zeros((1, Q.shape[0]))
        if (
            (S is not None) and S.shape[1] != Q.shape[0]
        ) or Q.shape[0] != Q.shape[1]:
            raise ValueError('Size mismatch')

    def jacobian(self, state, t):
        if state.ndim == 1:
            return 2 * (self.Q @ state).reshape(1, state.shape[0]) + self.S
        state_T = np.swapaxes(state, -1, -2)
        quadratic_term = self.Q @ state_T
        linear_term = self.S.reshape(1, -1)
        N = np.prod(state.shape[:-1])
        linear_term = np.vstack([linear_term] * N).reshape(state.shape)
        return quadratic_term + linear_term

    def get_measurement(self, state, *args, **kwargs):
        if state.ndim == 1:
            return state.T @ self.Q @ state + self.S @ state
        else:
            state_T = np.swapaxes(state, -1, -2)
            quadratic_term = (self.Q @ state_T) * state_T
            quadratic_term = quadratic_term.sum(axis=1)
            linear_term = self.S @ state_T
            quadratic_term = quadratic_term[..., np.newaxis]
            linear_term = np.swapaxes(linear_term, -1, -2)
        return quadratic_term + linear_term

