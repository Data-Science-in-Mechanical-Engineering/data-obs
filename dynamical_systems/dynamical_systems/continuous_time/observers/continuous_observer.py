from typing import Callable
from ...observer import Observer


class ContinuousTimeObserver(Observer):
    def __init__(self, dim, interpolation_method=None, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        self.interpolation_method = interpolation_method

    # Get observations = state estimations, result of observer
    # Takes t time steps at which to solve observer ODE, measurements and
    # controls vectors interpolated then used in observer
    # Returns solution of observer ODE
    def get_observations(self, t, measurements, controls=None, *args, **kwargs):
        raise NotImplementedError
