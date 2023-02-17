import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde

from .continuous_time import ContinuousTimeLTI
from .dynamical_system import ContinuousTimeSystem
from .state_initializer import GaussianInitializer
from .noise import LinearBrownianMotionNoise


# https://github.com/mattja/sdeint
# https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md
# https://gist.github.com/ChrisRackauckas/6a03e7b151c86b32d74b41af54d495c6


# Continuous-time SDE system, solved with torch to obtain discrete trajs
# SDE: dXt = f(Xt,t)dt + g(Xt,t)dWt
# Use def of g instead of noise property to define noise!!

# Need to create nn.Module for torchsde from functions f, g and switch from
# numpy to torch inside of it
class torchSDE(torch.nn.Module):
    def __init__(self, f, g, sde_type='ito', sde_noise_type='diagonal'):
        super().__init__()
        self.sde_type = sde_type
        self.noise_type = sde_noise_type
        self.f_np = f
        self.g_np = g

    def f(self, t, y):  # ugly but no choice?
        return torch.from_numpy(self.f_np(t.numpy(), y.numpy()))

    def g(self, t, y):  # ugly but no choice?
        return torch.from_numpy(self.g_np(t.numpy(), y.numpy()))


class TorchContinuousSystem(ContinuousTimeSystem):
    def __init__(self, dim, f, g, dt, sde_type='ito', sde_noise_type='diagonal',
                 solver_args={}, *args, **kwargs):
        super().__init__(dim, *args, **kwargs)
        # Create SDE object with torchsde
        self.sde = torchSDE(
            f=f, g=g, sde_type=sde_type, sde_noise_type=sde_noise_type)
        self.dt = dt  # length of time step for simulation
        self.solver_args = solver_args

    def get_trajectories(self, N_traj, T, t0=0., *args, **kwargs):
        tspan = torch.arange(t0, T, self.dt)
        x0 = np.zeros((N_traj, self.dim))
        for n_traj in range(N_traj):
            x0[n_traj] = self.state_initializer.initial_state()
        with torch.no_grad():
            traj = torch.transpose(torchsde.sdeint(
                sde=self.sde, y0=torch.from_numpy(x0), ts=tspan,
                **self.solver_args), 0, 1)
        return traj.numpy()


# Example with harmonic oscillator
if __name__ == '__main__':
    dim = 2
    dt = 0.1
    init1 = np.array([0., 1.])
    init2 = np.array([0., 2.])
    init_var = 1e-3
    noise_std = 1e-1
    Ntraj = 1
    T = 10

    start = time.time()

    A = np.array([[0, 1.], [-1, 0]])
    initializer_1 = GaussianInitializer(
        mean=init1, var=np.diag([init_var, init_var]))
    initializer_2 = GaussianInitializer(
        mean=init2, var=np.diag([init_var, init_var]))
    noise = LinearBrownianMotionNoise(sigma=noise_std*np.eye(dim))
    Syst = ContinuousTimeLTI(
        dim=dim, A=A, B=np.zeros_like(A), noise=noise, state_initializer=initializer_1
    )

    traj1, t1 = Syst.get_trajectories(N_traj=Ntraj, T=T, dt=dt)
    mid = time.time()

    Syst.state_initializer = initializer_2
    traj2, t2 = Syst.get_trajectories(N_traj=Ntraj, T=T, dt=dt)

    plt.plot(t1, traj1[0, :, 0], label=r'$x^1$')
    plt.plot(t1, traj1[0, :, 1], label=r'$\dot{x}^1$')
    plt.plot(t1, traj2[0, :, 0], label=r'$x^2$')
    plt.plot(t1, traj2[0, :, 1], label=r'$\dot{x}^2$')
    plt.legend()
    plt.show()

    mid1 = time.time()

    def f(t, x):
        A = np.array([[0, 1.], [-1, 0]])
        return np.matmul(A, x.T).T

    def g(t, x):
        return noise_std / np.sqrt(dt) * np.ones_like(x)
    torchSyst = TorchContinuousSystem(dim, f, g, dt,
                                      solver_args={'method': 'euler'})
    torchSyst.state_initializer = GaussianInitializer(
        mean=init1, var=np.array([init_var, init_var]))
    x1 = torchSyst.get_trajectories(N_traj=Ntraj, T=T)
    end = time.time()
    torchSyst.state_initializer = GaussianInitializer(
        mean=init2, var=np.array([init_var, init_var]))
    x2 = torchSyst.get_trajectories(N_traj=Ntraj, T=T)
    plt.plot(x1[0])
    plt.plot(x2[0])
    plt.show()

    print(x1.shape, traj1.shape)
    print(mid - start, end - mid1)
