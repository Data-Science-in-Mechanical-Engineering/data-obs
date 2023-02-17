import unittest
import matplotlib.pyplot as plt
import numpy as np

from dynamical_systems.continuous_time.systems import ContinuousTimeLTI
from dynamical_systems.continuous_time.observers import ContinuousTimeEKF
from dynamical_systems.noise.brownian_motions import LinearBrownianMotionNoise

from dynamical_systems.state_initializer import GaussianInitializer, ConstantInitializer
from dynamical_systems.noise import GaussianNoise
from dynamical_systems.measurement import LinearMeasurement, QuadraticFormMeasurement

from dynamical_systems.utils import set_seeds


set_seeds(0, 42)


def simulate_and_observe(system, observer, N_traj, T, dt):
    traj, t, controls = system.get_trajectories(
        N_traj=N_traj, T=T, dt=dt, return_controls=True)
    outputs = system.get_output_trajectories(N_traj, T, traj=traj)
    observations = observer.get_observations(t, outputs, controls)
    return traj, t, outputs, observations


class ContinuousEKFTest(unittest.TestCase):
    def run_test_1d(self, sys, obs, N_traj, T, dt):
        traj, t, outputs, observations = simulate_and_observe(
            sys, obs, N_traj, T, dt)
        for n_traj in range(N_traj):
            plt.plot(t, traj[n_traj, :, 0], color='blue', label='$x$')
            plt.plot(t, observations[n_traj, :, 0],
                     color='red', label='$\hat{x}$')
        plt.legend()
        plt.show()

    def run_test_2d(self, sys, obs, N_traj, T, dt):
        traj, t, outputs, observations = simulate_and_observe(
            sys, obs, N_traj, T, dt)
        for n_traj in range(N_traj):
            plt.plot(traj[n_traj, :, 0], traj[n_traj, :, 1])
            plt.plot(observations[n_traj, :, 0], observations[n_traj, :, 1])
        # plt.legend()
        # plt.xlim(traj.min(), traj.max())
        # plt.ylim(traj.min(), traj.max())
        plt.show()

    def test_1(self):
        N_traj = 1
        T = 10
        dt = 1e-2
        A = -1
        B = 0
        dim = 1
        sys = ContinuousTimeLTI(
            dim,
            A,
            B,
            state_initializer=GaussianInitializer(1, 0.1),
            meas_noise=GaussianNoise(0, 0.1),
        )
        obs = ContinuousTimeEKF(
            dim, system=sys, Q=0, state_initializer=ConstantInitializer(1)
        )
        self.run_test_1d(sys, obs, N_traj, T, dt)

    def test_2(self):
        N_traj = 2
        T = 10
        dt = 1e-2
        dim = 2
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        sys = ContinuousTimeLTI(
            dim,
            A,
            B,
            state_initializer=GaussianInitializer([1., 0.], 0.1),
            meas_noise=GaussianNoise([0., 0.], 0.001),
        )
        obs = ContinuousTimeEKF(
            dim, system=sys, Q=0, state_initializer=ConstantInitializer([0., 0.])
        )
        self.run_test_2d(sys, obs, N_traj, T, dt)

    def test_3(self):
        N_traj = 2
        T = 10
        dt = 1e-2
        dim = 2
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        noise_sigma = 0.1*np.eye(dim)
        sys = ContinuousTimeLTI(
            dim,
            A,
            B,
            state_initializer=GaussianInitializer([1., 0.], 0.1),
            meas_noise=GaussianNoise([0., 0.], 0.001),
            noise=LinearBrownianMotionNoise(sigma=noise_sigma)
        )
        obs = ContinuousTimeEKF(
            dim, system=sys, state_initializer=ConstantInitializer([0., 0.])
        )
        self.run_test_2d(sys, obs, N_traj, T, dt)

    def test_4(self):
        N_traj = 1
        T = 10
        dt = 1e-2
        dim = 2
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        measurement = LinearMeasurement(C=np.eye(1, 2))
        meas_noise = GaussianNoise([0.], 0.001)
        noise_sigma = 0.1*np.eye(dim)
        sys = ContinuousTimeLTI(
            dim,
            A,
            B,
            state_initializer=GaussianInitializer([1., 0.], 0.1),
            meas=measurement,
            meas_noise=meas_noise,
            noise=LinearBrownianMotionNoise(sigma=noise_sigma)
        )
        obs = ContinuousTimeEKF(
            dim, system=sys, state_initializer=ConstantInitializer([0., 0.])
        )
        self.run_test_2d(sys, obs, N_traj, T, dt)
