import unittest
import numpy as np

import matplotlib.pyplot as plt

from dynamical_systems.continuous_time import VanDerPol
from dynamical_systems.noise import LinearBrownianMotionNoise, NoNoise
from dynamical_systems.state_initializer import UniformInitializer

SEED = 0
RNG = np.random.default_rng(SEED)


class VanDerPolTest(unittest.TestCase):
    def create_system(self, mu):
        state_initializer = UniformInitializer([-5, -5], [5, 5])
        return VanDerPol(mu, state_initializer)

    def plot_traj(self, t, traj):
        plt.figure()
        for n, traj in enumerate(traj):
            plt.plot(traj[:, 0], traj[:, 1], label=f'Traj {n}')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    def test_1(self):
        noise = NoNoise()
        sys = self.create_system(mu=0)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=4, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    def test_2(self):
        noise = NoNoise()
        sys = self.create_system(mu=1)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=4, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    def test_3(self):
        # noise = LinearBrownianMotionNoise(np.array([[1.], [1.]]))
        noise = LinearBrownianMotionNoise(0.1*np.eye(2))
        sys = self.create_system(mu=0)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=4, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    def test_4(self):
        # noise = LinearBrownianMotionNoise(np.array([[1.], [1.]]))
        noise = LinearBrownianMotionNoise(0.1*np.eye(2))
        sys = self.create_system(mu=1)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=4, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    # def test_2(self):
    #     dim_x = 1
    #     noise = LinearBrownianMotionNoise(np.eye(dim_x))
    #     sys = self.create_system(
    #         dim_x=dim_x, pole_placement_regularizer_abs=1)
    #     sys.noise = noise
    #     traj, t = sys.get_trajectories(N_traj=4, T=100, dt=1e-2)
    #     self.plot_traj(t, traj)
