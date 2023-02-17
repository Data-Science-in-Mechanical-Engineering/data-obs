import unittest
import numpy as np

import matplotlib.pyplot as plt

from dynamical_systems.continuous_time import Duffing
from dynamical_systems.noise import LinearBrownianMotionNoise, NoNoise
from dynamical_systems.controller import SinusoidalController
from dynamical_systems.state_initializer import UniformInitializer

SEED = 0
RNG = np.random.default_rng(SEED)


class DuffingTest(unittest.TestCase):
    def create_system(self, alpha, beta, delta):
        state_initializer = UniformInitializer([-1, -1], [1, 1])
        return Duffing(alpha, beta, delta, state_initializer)

    def plot_traj(self, t, traj):
        plt.figure()
        for n, traj in enumerate(traj):
            plt.plot(traj[:, 0], traj[:, 1], label=f'Traj {n}')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def test_1(self):
        noise = NoNoise()
        sys = self.create_system(alpha=-1, beta=1, delta=0)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=10, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    def test_2(self):
        noise = LinearBrownianMotionNoise(0.1*np.eye(2))
        sys = self.create_system(alpha=-1, beta=1, delta=0)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=10, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    def test_3(self):
        noise = NoNoise()
        controller = SinusoidalController(dim=1, amplitude=0.5, pulse=1.2)
        sys = self.create_system(alpha=-1, beta=1, delta=0.3)
        sys.noise = noise
        sys.controller = controller
        traj, t = sys.get_trajectories(N_traj=4, T=60, dt=1e-2)
        self.plot_traj(t, traj)

    def test_4(self):
        # noise = LinearBrownianMotionNoise(np.array([[1.], [1.]]))
        noise = LinearBrownianMotionNoise(0.1*np.eye(2))
        controller = SinusoidalController(dim=1, amplitude=0.5, pulse=1.2)
        sys = self.create_system(alpha=-1, beta=1, delta=0.3)
        sys.noise = noise
        sys.controller = controller
        traj, t = sys.get_trajectories(N_traj=4, T=60, dt=1e-2)
        self.plot_traj(t, traj)