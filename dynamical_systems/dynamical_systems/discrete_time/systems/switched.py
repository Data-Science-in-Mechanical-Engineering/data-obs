import numpy as np

from ...dynamical_system import DiscreteTimeSystem


# Careful: don't use the same SwitchedSystem instance in two different places
# when switch_type is AUTO_SWITCH: it relies on an internal state!
class SwitchedSystem(DiscreteTimeSystem):
    AUTO_SWITCH = 'auto switch'
    MANUAL_SWITCH = 'manual switch'

    S1 = 's1'
    S2 = 's2'

    def __init__(self, s1, s2, *args, switch_time=None, **kwargs):
        if s1.dim != s2.dim:
            raise ValueError(
                'The two systems must have the same state dimension')
        dim = s1.dim
        super().__init__(dim, *args, **kwargs)

        self.s1 = s1
        self.s2 = s2

        self.switch_time = switch_time
        self.internal_t = 0
        self.switch_type = None
        self.reset_switch_type(None)
        self.manual_switch = SwitchedSystem.S1

    def switch(self, to=None):
        self.switch_type = SwitchedSystem.MANUAL_SWITCH
        if to is None:
            to = SwitchedSystem.S1 if self.manual_switch == SwitchedSystem.S2 else SwitchedSystem.S2
        self.manual_switch = to

    def reset_switch_type(self, switch_type=None):
        if switch_type is None:
            switch_type = SwitchedSystem.AUTO_SWITCH \
                if self.switch_time is not None \
                else SwitchedSystem.MANUAL_SWITCH
        if switch_type == SwitchedSystem.AUTO_SWITCH and self.switch_time is None:
            raise ValueError(
                "Cannot switch to AUTO_SWITCH if switch_time is None")
        self.switch_type = switch_type

    @property
    def s(self):
        if (
            self.switch_type == SwitchedSystem.MANUAL_SWITCH and
            self.manual_switch == SwitchedSystem.S1
        ) or (
            self.switch_type == SwitchedSystem.AUTO_SWITCH and
            self.internal_t < self.switch_time
        ):
            return self.s1
        else:
            return self.s2

    @property
    def controller(self):
        return self.s.controller

    @controller.setter
    def controller(self, controller):
        self.s.controller = controller

    @property
    def state_initializer(self):
        return self.s.state_initializer

    @state_initializer.setter
    def state_initializer(self, state_initializer):
        self.s.state_initializer = state_initializer

    @property
    def noise(self):
        return self.s.noise

    @noise.setter
    def noise(self, noise):
        self.s.noise = noise

    def get_next_state_control(self, state, t, *args, **kwargs):
        u = self.controller.get_control_input(state, t)
        next_state, _ = self.s.get_next_state_control(
            state, t, *args, **kwargs)
        self.internal_t += 1
        return next_state, u

    def get_trajectories(self, N_traj, T, *args, **kwargs):
        traj = np.zeros((N_traj, T, self.dim))
        for n_traj in range(N_traj):
            self.internal_t = 0
            traj[n_traj, 0, :] = self.state_initializer.initial_state()
            for t in range(T-1):
                traj[n_traj, t +
                     1] = self.get_next_state_control(traj[n_traj, t], *args, **kwargs)
                self.internal_t = t
        self.internal_t = 0
        return traj
