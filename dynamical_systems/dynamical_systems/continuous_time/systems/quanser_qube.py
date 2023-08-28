import numpy as np
import torch
import copy
from functorch import vmap, jacrev

from ...dynamical_system import ContinuousTimeSystem


# Simulation model for Quanser Qube 2
# In pytorch instead of usual numpy to enable autodiff for EKF (ugly...)

class QuanserQubeServo2(ContinuousTimeSystem):
    """ See https://www.quanser.com/products/qube-servo-2/ QUBE SERVO 2 and
    for a detailed reference for this system.
    Documentation on the simulator:
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/gym_brt/quanser/qube_simulator.py
    https://github.com/BlueRiverTech/quanser-openai-driver/blob/main/tests/notebooks/Compare%20Qube%20Hardware%20to%20ODEint.ipynb.

    State: (theta, alpha, theta_dot, alpha_dot)
    Measurement: (theta, alpha)
    """

    def __init__(self, state_initializer, physical_parameters=None, *args, **kwargs):
        super().__init__(dim=4, state_initializer=state_initializer, *args, **kwargs)

        if physical_parameters is None:
            physical_parameters = {}

        # Motor
        # self.Rm = physical_parameters.get('Rm', 8.4)  # Resistance
        self.kt = physical_parameters.get(
            'kt', 0.042)  # Current-torque (N-m/A)
        # self.km = physical_parameters.get(
        #     'km', 0.042)  # Back-emf constant (V-s/rad)
        # Rotary Arm
        self.mr = physical_parameters.get('mr', 0.095)  # Mass (kg)
        self.Lr = physical_parameters.get('Lr', 0.085)  # Total length (m)
        # Moment of inertia about pivot (kg-m^2)
        self.Jr = physical_parameters.get('Jr', self.mr * self.Lr ** 2 / 12)
        # Equivalent viscous damping coefficient (N-m-s/rad)
        # self.Dr = physical_parameters.get('Dr', 0.00027)
        # Pendulum Link
        self.mp = physical_parameters.get('Mp', 0.024)  # Mass (kg)
        self.Lp = physical_parameters.get('Lp', 0.129)  # Total length (m)
        # Moment of inertia about pivot (kg-m^2)
        self.Jp = physical_parameters.get('Jp', self.mp * self.Lp ** 2 / 12)
        # Equivalent viscous damping coefficient (N-m-s/rad)
        # self.Dp = physical_parameters.get('Dp', 0.00005)

        # After identification on hardware data:
        self.Rm = physical_parameters.get('Rm', 14)
        self.km = physical_parameters.get('km', 0.01)
        self.Dr = physical_parameters.get('Dr', 0.0005)
        self.Dp = physical_parameters.get('Dp', -3e-5)

        self.gravity = physical_parameters.get(
            'gravity', 9.81)  # Gravity constant (m/s^2)

    def f_torch(self, x):
        theta = x[..., 0]
        alpha = x[..., 1]
        theta_dot = x[..., 2]
        alpha_dot = x[..., 3]

        Vm = 0.  # action/control input
        # Not handled so far, since it requires to implement controllers in PyTorch
        tau = -(self.km * (Vm - self.km * theta_dot)) / self.Rm

        xdot = torch.zeros_like(x)
        xdot[..., 0] = theta_dot
        xdot[..., 1] = alpha_dot
        xdot[..., 2] = (
            -self.Lp * self.Lr * self.mp * (
                -8.0 * self.Dp * alpha_dot + self.Lp ** 2 * self.mp *
                theta_dot ** 2 * torch.sin(2.0 * alpha) + 4.0 * self.Lp *
                self.gravity * self.mp * torch.sin(alpha)
            ) * torch.cos(alpha) + (
                4.0 * self.Jp + self.Lp ** 2 * self.mp
            ) * (
                4.0 * self.Dr * theta_dot + self.Lp ** 2 *
                alpha_dot * self.mp * theta_dot * torch.sin(
                    2.0 * alpha
                ) + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 *
                self.mp * torch.sin(alpha) - 4.0 * tau
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2
            * torch.cos(alpha) ** 2 - (
                4.0 * self.Jp + self.Lp ** 2 * self.mp
            ) * (
                4.0 * self.Jr + self.Lp ** 2 * self.mp *
                torch.sin(alpha) ** 2 + 4.0 * self.Lr ** 2 *
                self.mp
            )
        )

        xdot[..., 3] = (
            2.0 * self.Lp * self.Lr * self.mp * (
                4.0 * self.Dr * theta_dot + self.Lp ** 2 * alpha_dot
                * self.mp * theta_dot * torch.sin(2.0 * alpha)
                + 2.0 * self.Lp * self.Lr * alpha_dot ** 2 * self.mp
                * torch.sin(alpha) - 4.0 * tau
            ) * torch.cos(alpha) - 0.5 * (
                4.0 * self.Jr + self.Lp ** 2 * self.mp *
                torch.sin(alpha) ** 2 + 4.0 * self.Lr ** 2 * self.mp
            ) * (
                -8.0 * self.Dp * alpha_dot + self.Lp ** 2 * self.mp
                * theta_dot ** 2 * torch.sin(2.0 * alpha)
                + 4.0 * self.Lp * self.gravity * self.mp
                * torch.sin(alpha)
            )
        ) / (
            4.0 * self.Lp ** 2 * self.Lr ** 2 * self.mp ** 2
            * torch.cos(alpha) ** 2
            - (4.0 * self.Jp + self.Lp ** 2 * self.mp)
            * (
                4.0 * self.Jr + self.Lp ** 2 * self.mp *
                torch.sin(alpha) ** 2 + 4.0 * self.Lr ** 2
                * self.mp
            )
        )

        return xdot

    def f(self, x, t):
        x = torch.as_tensor(x)
        t = torch.as_tensor(t)
        xdot = self.f_torch(x)
        return xdot.numpy()

    # For flexibility and coherence: use remap function after every simulation
    # But be prepared to change its behavior!
    @staticmethod
    def remap_angles(traj):
        # Map theta to [-pi,pi] and alpha to [0, 2pi]
        traj[..., 0] = ((traj[..., 0] + np.pi) % (2 * np.pi)) - np.pi
        traj[..., 1] = traj[..., 1] % (2 * np.pi)
        return traj

    # For adapting hardware data to the conventions of the simulation model
    @staticmethod
    def remap_hardware_angles(traj, add_pi_alpha=True, mod=False):
        # Reorder as (theta, alpha, thetadot, alphadot)
        # Convention for alpha: 0 is upwards (depends on dataset!)
        # Remap as simulation data
        traj_copy = copy.deepcopy(traj)
        traj[..., 0], traj[..., 1] = traj_copy[..., 1], traj_copy[..., 0]
        traj[..., 2], traj[..., 3] = traj_copy[..., 3], traj_copy[..., 2]
        if mod:
            # Map theta to [-pi,pi] and alpha to [0, 2pi]
            traj[..., 0] = ((traj[..., 0] + np.pi) % (2 * np.pi)) - np.pi
            traj[..., 1] = traj[..., 1] % (2 * np.pi)
        if add_pi_alpha:
            traj[..., 1] += np.pi
        # return QuanserQubeServo2.remap_angles(traj)
        return traj

    def get_trajectories(self, N_traj, T, *args, **kwargs):
        traj, t = super().get_trajectories(
            N_traj=N_traj, T=T, *args, **kwargs)
        # return self.remap_angles(traj), t
        return traj, t

    def __repr__(self):
        return "QuanserQubeServo2"

    def jacobian(self, state, control, t):
        jac = self.predict_deriv_torch(torch.as_tensor(state), self.f_torch)
        return jac.numpy()

    # From Mona's EKF implementation
    # Functions to interface with dynamical_systems library are defined above
    # Useful functions for EKF

    def __call__(self, t, x, u, t0, init_control, process_noise_var, kwargs,
                 impose_init_control=False):
        return self.f(x, t)

    def call_deriv(self, t, x, u, t0, init_control, process_noise_var,
                   kwargs, impose_init_control=False):
        return self.predict_deriv_torch(torch.as_tensor(x), self.f).numpy()

    def call_deriv_torch(self, t, x, u, t0, init_control, process_noise_var,
                         kwargs, impose_init_control=False):
        return self.predict_deriv_torch(x, self.f)

    # Jacobian of function that predicts xdot from x. Useful for EKF!
    def predict_deriv_torch(self, x, f):
        # Compute Jacobian of f with respect to input x
        if len(x.shape) > 1:
            dfdh = vmap(jacrev(f))(x)
        else:
            dfdh = jacrev(f)(x)
        dfdx = torch.squeeze(dfdh)
        return dfdx
    
    # Check observability analytically with both Kalman criterion and Lie 
    # observability matrix, at reference point x with observation vector C
    def check_observability(self, x, C):
        x = torch.as_tensor(x)
        C = torch.as_tensor(C)

        def obs_vect1(x):
            return torch.matmul(C, x.t())

        def obs_vect2(x):
            f = self.f_torch(x)
            return torch.matmul(C, f.t())

        def obs_vect3(x):
            f = self.f_torch(x)
            J = torch.squeeze(jacrev(self.f_torch)(x))
            return torch.matmul(C, torch.matmul(J, f.t()))

        def obs_vect4(x):
            f = self.f_torch(x)
            H = torch.squeeze(jacrev(obs_vect3)(x))
            return torch.matmul(H, f.t())

        J = torch.squeeze(jacrev(self.f_torch)(x))
        K = torch.vstack((C,
                          torch.matmul(C, J),
                          torch.matmul(C, torch.matmul(J, J)),
                          torch.matmul(C, torch.matmul(J, torch.matmul(J, J)))
                          ))
        print('Kalman observability criterion: rank',
              torch.linalg.matrix_rank(K))
        J1 = torch.squeeze(vmap(jacrev(obs_vect1))(x))
        J2 = torch.squeeze(vmap(jacrev(obs_vect2))(x))
        J3 = torch.squeeze(vmap(jacrev(obs_vect3))(x))
        J4 = torch.squeeze(vmap(jacrev(obs_vect4))(x))
        O = torch.vstack((J1, J2, J3, J4))
        print('Lie observability matrix: rank',
              torch.linalg.matrix_rank(O))
