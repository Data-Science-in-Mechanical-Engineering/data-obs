import numpy as np

from ...dynamical_system import DiscreteTimeSystem


class BakersMap(DiscreteTimeSystem):
    def __init__(self, folded=False, *args, **kwargs):
        super().__init__(dim=2, *args, **kwargs)
        self.folded = folded

    def get_next_state_control(self, state, t, *args, **kwargs):
        u = self.controller.get_control_input(state, t)
        x, y = np.squeeze(state)
        if not self.folded:
            next_state = np.array([
                (2*x) % 1, (y + int(2*x))/2
            ])
        else:
            next_state = np.array([
                2*x, y/2
            ]) if x < 1/2 else np.array([
                2 - 2*x, 1 - y/2
            ])
        return next_state, u
