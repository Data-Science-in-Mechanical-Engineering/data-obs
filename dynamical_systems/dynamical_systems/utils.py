import numpy as np
import torch
from collections.abc import Iterable
from typing import Callable
import scipy.interpolate


class RNG:
    """
    Singleton class for random number generation in this package
    Enables reproducibility

    WARNING: setting the seed/RNG here has no influence over `sdeint`,
        you should still use np.random.seed for this, or the function set_seeds
        in this module
    """
    __RNG = None

    def __init__(self):
        raise NotImplementedError(
            'RNG cannot be instantiated; use RNG.get() instead.'
        )

    @classmethod
    def set_rng(cls, new_rng):
        if isinstance(new_rng, np.random.Generator):
            cls.__RNG = new_rng
        else:
            raise ValueError(
                f'new_rng can only be a np.random.Generator, not a {type(new_rng)}'
            )

    @classmethod
    def set_seed(cls, seed):
        cls.__RNG = np.random.default_rng(seed=seed)

    @classmethod
    def get(cls) -> np.random.Generator:
        if cls.__RNG is None:
            cls.set_seed(seed=None)
        return cls.__RNG


def set_seeds(seed_np, seed_rng=None):
    """
    Sets seeds for all RNGs used in this package
    The np.random.seed is only used in sdeint
    """
    np.random.seed(seed_np)
    RNG.set_seed(seed=seed_rng if seed_rng is not None else seed_np)


def identity_or_broadcast(to_return_or_broadcast, dim, matrix=True):
    if isinstance(to_return_or_broadcast, np.ndarray):
        return to_return_or_broadcast
    elif isinstance(to_return_or_broadcast, Iterable):
        to_return_or_broadcast = np.atleast_2d(
            to_return_or_broadcast) if matrix else np.atleast_1d(to_return_or_broadcast)
    else:
        broadcast = np.eye(dim) if matrix else np.ones(dim)
        to_return_or_broadcast = to_return_or_broadcast * broadcast
    return to_return_or_broadcast


# Reshape any vector of (length,) to (1, length) (single point of certain
# dimension)
def reshape_pt1(x):
    if np.isscalar(x) or np.array(x).ndim == 0:
        x = np.reshape(x, (1, 1))
    else:
        x = np.array(x)
    if len(x.shape) == 1:
        x = np.reshape(x, (1, x.shape[0]))
    return x


# Takes as input t time steps at which meas_traj is known
# Returns a function interpolation which will take a vector t_new as
# input and interpolate meas_traj at time steps in t_new
def interpolate_trajectory(t, meas_traj, kind=None):
    if isinstance(meas_traj, Callable):
        return meas_traj

    else:
        interpolation = scipy.interpolate.interp1d(
            x=t,
            y=meas_traj,
            axis=0,
            kind='linear' if kind is None else kind
        )
        return interpolation
