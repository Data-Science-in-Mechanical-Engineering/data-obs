import unittest
import numpy as np

from dynamical_systems.noise import GaussianNoise
from dynamical_systems.utils import RNG


class RNGTest(unittest.TestCase):
    def test_1(self):
        self.assertRaises(NotImplementedError, RNG)

    def test_2(self):
        self.assertTrue(isinstance(RNG.get(), np.random.Generator))

    def test_3(self):
        RNG.set_seed(0)
        r1 = RNG.get().uniform(0, 1)
        RNG.set_seed(0)
        r2 = RNG.get().uniform(0, 1)
        self.assertAlmostEqual(r1, r2)

    def test_4(self):
        default_rng = np.random.default_rng(seed=0)
        r1 = default_rng.uniform(0, 1)
        RNG.set_seed(0)
        r2 = RNG.get().uniform(0, 1)
        self.assertAlmostEqual(r1, r2)

    def test_5(self):
        rng = np.random.default_rng()
        RNG.set_rng(rng)
        self.assertEqual(rng, RNG.get())

    def test_6(self):
        default_rng = np.random.default_rng(seed=0)
        r1 = default_rng.uniform(0, 1)
        rng = np.random.default_rng(seed=0)
        RNG.set_rng(rng)
        r2 = RNG.get().uniform(0, 1)
        self.assertAlmostEqual(r1, r2)

    def test_7(self):
        rng1 = np.random.default_rng(0)
        RNG.set_rng(rng1)
        r1 = rng1.random()
        r2 = RNG.get().random()

        rng2 = np.random.default_rng(0)
        s1 = rng2.random()
        s2 = rng2.random()

        self.assertAlmostEqual(r1, s1)
        self.assertAlmostEqual(r2, s2)
