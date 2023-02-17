import numpy as np
from ..utils import RNG, identity_or_broadcast


class Noise:
    def __init__(self, *args, **kwargs):
        pass

    def get_noise_input(self, states, *args, **kwargs):
        raise NotImplementedError


class NoNoise(Noise):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_noise_input(self, states, *args, **kwargs):
        return np.zeros_like(states)


class GaussianNoise(Noise):
    def __init__(self, mean, var, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = np.atleast_1d(mean)
        self.var = identity_or_broadcast(var, self.mean.shape[0])
        if (self.mean.shape + self.mean.shape) != self.var.shape:
            raise ValueError(
                f'Size mismatch: mean has shape {self.mean.shape}, '
                f'and variance has shape {self.var.shape}'
            )

    def get_noise_input(self, states, *args, **kwargs):
        size = states.shape[:-1]
        return RNG.get().multivariate_normal(self.mean, self.var, size=size)
