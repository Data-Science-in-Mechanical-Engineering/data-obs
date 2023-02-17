import unittest
import matplotlib.pyplot as plt
import numpy as np

from dynamical_systems.continuous_time import ContinuousTimeLTI
from dynamical_systems.noise import LinearBrownianMotionNoise
from dynamical_systems.state_initializer import GaussianInitializer, ConstantInitializer
from dynamical_systems.measurement import LinearMeasurement, QuadraticFormMeasurement
from dynamical_systems.continuous_time.observers import ContinuousTimeLuenberger

from dynamical_systems.utils import set_seeds


set_seeds(0, 42)


def simulate_and_observe(system, observer, N_traj, T, dt):
    traj, t, controls = system.get_trajectories(
        N_traj=N_traj, T=T, dt=dt, return_controls=True)
    outputs = system.get_output_trajectories(N_traj, T, traj=traj)
    observations = observer.get_observations(t, outputs, controls)
    return traj, t, outputs, observations


class ContinuousLuenbergerTest(unittest.TestCase):
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
        plt.xlim(traj.min(), traj.max())
        plt.ylim(traj.min(), traj.max())
        plt.show()

    def test_1(self):
        N_traj = 1
        T = 10
        dt = 1e-2
        A = -1
        B = 0
        dim = 1
        sys = ContinuousTimeLTI(dim, A, B, ConstantInitializer(1))
        obs = ContinuousTimeLuenberger(dim, L=1, system=sys)
        self.run_test_1d(sys, obs, N_traj, T, dt)

    def test_2(self):
        N_traj = 4
        T = 10
        dt = 1e-2
        A = -1
        B = 0
        dim = 1
        sys = ContinuousTimeLTI(dim, A, B, GaussianInitializer(1, 1))
        obs = ContinuousTimeLuenberger(dim, L=1, system=sys)
        self.run_test_1d(sys, obs, N_traj, T, dt)

    def test_3(self):
        N_traj = 4
        T = 1
        dt = 1e-2
        dim = 2
        meas_dim = dim
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        L = 10*np.eye(N=dim, M=meas_dim)
        sys = ContinuousTimeLTI(
            dim, A, B, GaussianInitializer(np.zeros(dim), np.eye(dim)))
        obs = ContinuousTimeLuenberger(dim, L=L, system=sys)
        self.run_test_2d(sys, obs, N_traj, T, dt)

    def test_4(self):
        N_traj = 4
        T = 5
        dt = 1e-2
        dim = 2
        meas_dim = dim
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        noise_sigma = 0.1*np.eye(dim)
        L = 10*np.eye(N=dim, M=meas_dim)
        sys = ContinuousTimeLTI(
            dim,
            A=A,
            B=B,
            state_initializer=GaussianInitializer(np.zeros(dim), np.eye(dim)),
            noise=LinearBrownianMotionNoise(noise_sigma)
        )
        obs = ContinuousTimeLuenberger(dim, L=L, system=sys)
        self.run_test_2d(sys, obs, N_traj, T, dt)

    def test_5(self):
        N_traj = 1
        T = 5
        dt = 1e-2
        dim = 2
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        noise_sigma = 0.1*np.eye(dim)
        measurement = LinearMeasurement(C=np.eye(1, 2))
        L = 2*np.ones((dim, measurement.dim))
        sys = ContinuousTimeLTI(
            dim,
            A=A,
            B=B,
            state_initializer=GaussianInitializer(np.zeros(dim), np.eye(dim)),
            # noise=LinearBrownianMotionNoise(noise_sigma),
            meas=measurement
        )
        obs = ContinuousTimeLuenberger(dim, L=L, system=sys)
        self.run_test_2d(sys, obs, N_traj, T, dt)

    def test_6(self):
        N_traj = 4
        T = 5
        dt = 1e-2
        dim = 2
        A = np.array([[0., 1.], [-1., 0.]])
        B = np.zeros((dim, 1))
        noise_sigma = 0.1*np.eye(dim)
        measurement = QuadraticFormMeasurement(Q=np.eye(2))
        L = np.array([0., 1.]).reshape(-1, 1)
        sys = ContinuousTimeLTI(
            dim,
            A=A,
            B=B,
            state_initializer=GaussianInitializer(np.zeros(dim), np.eye(dim)),
            # noise=LinearBrownianMotionNoise(noise_sigma),
            meas=measurement
        )
        obs = ContinuousTimeLuenberger(dim, L=L, system=sys)
        self.run_test_2d(sys, obs, N_traj, T, dt)
