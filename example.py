import numpy as np

from kernels_dynamical_systems.custom_kernels import TrajectoryRBFTwoSampleTest
from dynamical_systems.dynamical_systems import LTISystem, GaussianInitializer, GaussianNoise, \
    DLQRController, set_seeds, RNG

# Example of dynamical system and kernel2sample test

seed_np = 1
seed_rng = 2
set_seeds(seed_np, seed_rng)

dim = 2

A1 = RNG.get().uniform(size=(dim, dim))
A2 = RNG.get().uniform(size=(dim, dim))
B = np.eye(dim)
Q = np.eye(dim)
R = 10 ** 10 * np.eye(dim)

noise = GaussianNoise(0, 0.1)
initializer = GaussianInitializer(0, 1)

# Define two different systems

s1 = LTISystem(
    dim, A1, B,
    state_initializer=initializer,
    noise=noise,
    controller=DLQRController(A1, B, Q, R),
)

s2 = LTISystem(
    dim, A2, B,
    state_initializer=initializer,
    noise=noise,
    controller=DLQRController(A2, B, Q, R),
)

# Create three data sets

N11, N12, N2 = 300, 300, 300
T = 2
print('Generating data...')
data11 = s1.get_trajectories(N_traj=N11, T=T)
data12 = s1.get_trajectories(N_traj=N12, T=T)
data2 = s2.get_trajectories(N_traj=N2, T=T)
print('Done.')

# Test for the difference between these data sets

alpha = 0.05  # confidence threshold
sigma = None  # Bandwidth of the RBF kernel. `None` computes the median distance of the data set

test11_12 = TrajectoryRBFTwoSampleTest(X=data11, Y=data12, alpha=alpha,
                                       sigma=sigma)
test11_2 = TrajectoryRBFTwoSampleTest(X=data11, Y=data2, alpha=alpha,
                                      sigma=sigma)
test12_2 = TrajectoryRBFTwoSampleTest(X=data12, Y=data2, alpha=alpha,
                                      sigma=sigma)

# Run the tests
print("Results:")
for test, truth, labels in zip([test11_12, test11_2, test12_2],
                               [True, False, False],
                               [('11', '12'), ('11', '2'), ('12', '2')]):
    print('==================')
    print(f'Test {labels[0]} vs. {labels[1]}')
    test.perform_test(verbose=True)
    result = test.is_null_hypothesis_accepted()
    print(f'The outcome is {"CORRECT" if result == truth else "INCORRECT"}')
print('Done.')
