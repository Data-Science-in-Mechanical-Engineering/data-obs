import numpy as np
from .utils import RNG, identity_or_broadcast


class StateInitializer:
    def __init__(self, *args, **kwargs):
        pass

    def initial_state(self, size=None, *args, **kwargs):
        raise NotImplementedError


class UniformInitializer(StateInitializer):
    def __init__(self, bounds_inf, bounds_sup, *args, **kwargs):
        if len(bounds_inf) != len(bounds_sup):
            raise ValueError(
                'There should be the same number of inferior and superior bounds')

        super().__init__(*args, **kwargs)

        self.bounds_inf = bounds_inf
        self.bounds_sup = bounds_sup

    def initial_state(self, size=None, *args, **kwargs):
        return RNG.get().uniform(self.bounds_inf, self.bounds_sup, size=size)


class GaussianInitializer(StateInitializer):
    def __init__(self, mean, var, *args, **kwargs):
        self.mean = np.atleast_1d(mean)
        self.var = identity_or_broadcast(var, self.mean.shape[0])
        if (self.mean.shape + self.mean.shape) != self.var.shape:
            raise ValueError(
                f'Size mismatch: mean has shape {self.mean.shape}, '
                f'and variance has shape {self.var.shape}'
            )

    def initial_state(self, size=None, *args, **kwargs):
        return RNG.get().multivariate_normal(self.mean, self.var, size=size)


class ConstantInitializer(StateInitializer):
    def __init__(self, state, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = np.atleast_1d(state)

    def initial_state(self, size=None, *args, **kwargs):
        if size is None:
            return self.state
        N = np.prod(size)
        state_to_stack = self.state[np.newaxis, ...]
        initial_states = np.vstack([state_to_stack for _ in range(N)])
        return initial_states
