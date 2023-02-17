import numpy as np

from .state_initializer import ConstantInitializer


class Observer:
    def __init__(self, dim, state_initializer=None, *args, **kwargs):
        self.dim = dim
        self.state_initializer = state_initializer if state_initializer is not None \
            else ConstantInitializer(np.zeros(dim, dtype=float))

    def get_observations(self, t, measurements, controls=None, *args, **kwargs):
        raise NotImplementedError()


class NoObserver(Observer):
    def __init__(self, dim, state_initializer=None, *args, **kwargs):
        super().__init__(dim=dim, state_initializer=None, *args, **kwargs)

    def get_observations(self, measurements, controls, t, *args, **kwargs):
        N = measurements.shape[0]
        T = t.shape[0]
        return self.state_initializer(size=(N, T))
