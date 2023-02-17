from ...dynamical_system import DiscreteTimeSystem


class LTISystem(DiscreteTimeSystem):
    def __init__(self, dim, A, B, state_initializer, *args, **kwargs):
        super().__init__(dim, state_initializer=state_initializer, *args, **kwargs)
        self.A = A
        self.B = B

        dim_match = (A.shape == (dim, dim)) and (
            B.shape[0] == dim)
        if not dim_match:
            raise ValueError("Dimension mismatch")

    def get_next_state_control(self, state, t, *args, **kwargs):
        u = self.controller.get_control_input(state, t)
        noise = self.noise.get_noise_input(state)
        next_state = self.A @ state + self.B @ u + noise
        return next_state, u
