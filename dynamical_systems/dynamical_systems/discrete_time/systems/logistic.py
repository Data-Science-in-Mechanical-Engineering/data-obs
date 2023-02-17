import numpy as np

from ...dynamical_system import DiscreteTimeSystem


class LogisticSystem(DiscreteTimeSystem):
    def __init__(self, mu, *args, **kwargs):
        super().__init__(dim=1, *args, **kwargs)
        self.mu = mu

    def get_next_state_control(self, state, t, *args, **kwargs):
        noise = self.noise.get_noise_input(state)
        u = self.controller.get_control_input(state, t)
        next_state = np.clip(self.mu * state * (1 - state) + noise, 0, 1)
        return next_state, u
