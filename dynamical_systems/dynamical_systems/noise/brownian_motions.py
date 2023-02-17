from .noise import Noise


class BrownianMotionNoise(Noise):
    def G(self, state, time, *args, **kwargs):
        raise NotImplementedError

    def get_noise_input(self, states, *args, **kwargs):
        raise NotImplementedError(
            'This class is intended as being used in an SDE with a ContinuousTimeSystem')


class LinearBrownianMotionNoise(BrownianMotionNoise):
    def __init__(self, sigma, *args, **kwargs):
        self.sigma = sigma
        super().__init__(*args, **kwargs)

    def G(self, state, time, *args, **kwargs):
        return self.sigma
