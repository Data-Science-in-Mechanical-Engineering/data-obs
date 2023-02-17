import unittest
import numpy as np

import matplotlib.pyplot as plt

from dynamical_systems.continuous_time import ContinuousTimeLTI
from dynamical_systems.noise import LinearBrownianMotionNoise, NoNoise
from dynamical_systems.state_initializer import GaussianInitializer

SEED = 0
RNG = np.random.default_rng(SEED)


class ContinuousTimeLTISystemTest(unittest.TestCase):
    def create_system(self, dim_x, pole_placement_regularizer_rel=1.5, pole_placement_regularizer_abs=1.):
        A = RNG.normal(size=(dim_x, dim_x))
        eigvals = np.linalg.eigvals(A)
        max_eigval = np.max(eigvals)
        regularizer = (pole_placement_regularizer_abs +
                       (max_eigval > 0) * pole_placement_regularizer_rel * max_eigval)
        A = A - regularizer * np.eye(dim_x)
        # A now has negative eigenvalues; the system is stable
        state_initializer = GaussianInitializer(
            np.zeros(dim_x), 10*np.eye(dim_x))
        sys = ContinuousTimeLTI(
            dim=dim_x, state_initializer=state_initializer, A=A, B=np.atleast_2d(0.))
        return sys

    def plot_traj(self, t, traj):
        plt.figure()
        plt.plot(t, traj.squeeze().T)
        plt.show()

    def test_1(self):
        dim_x = 1
        noise = NoNoise()
        sys = self.create_system(dim_x=dim_x)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=4, T=10, dt=1e-2)
        self.plot_traj(t, traj)

    def test_2(self):
        dim_x = 1
        noise = LinearBrownianMotionNoise(np.eye(dim_x))
        sys = self.create_system(
            dim_x=dim_x, pole_placement_regularizer_abs=1)
        sys.noise = noise
        traj, t = sys.get_trajectories(N_traj=4, T=100, dt=1e-2)
        self.plot_traj(t, traj)
