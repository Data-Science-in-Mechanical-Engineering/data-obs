import unittest
import numpy as np

from dynamical_systems.noise import GaussianNoise


class GaussianNoiseTest(unittest.TestCase):
    def test_1(self):
        noise = GaussianNoise(0, 1)
        outcome = noise.get_noise_input(np.array([0]))
        self.assertEqual(outcome.shape, (1,))

    def test_2(self):
        noise = GaussianNoise([0, 0], 2)
        self.assertTrue(np.all(noise.mean == np.array([0, 0])))
        self.assertTrue(np.all(noise.var == 2*np.eye(2)))

    def test_3(self):
        self.assertRaises(
            ValueError,
            GaussianNoise,
            [0., 0.],
            [1.]
        )

    def test_4(self):
        self.assertRaises(
            ValueError,
            GaussianNoise,
            [0.],
            [[1., 2.], [2., 1.]]
        )
