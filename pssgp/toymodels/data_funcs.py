"""
Some toy models based on deterministic test functions
"""
import math
from typing import Optional

import numpy as np


def sinu(t: np.ndarray) -> np.ndarray:
    """
    A sinusoidal test function. 
    
    y = sin(pi t) + sin(2 pi t) + cos(3 pi t)
    
    Args:
        t:  (n, ) Input (time)
    Return:
        y:  (n, ) y(t)
    """
    return np.sin(np.pi * t) \
           + np.sin(2 * np.pi * t) \
           + np.cos(3 * np.pi * t)


def comp_sinu(t: np.ndarray) -> np.ndarray:
    """
    A composite sinusoidal test function. It is very
    challenging for statitionary GP to model.
    
    y = sin^2(7 pi cos(2 pi t^2) t) / (cos(t pi t) + 2)
    
    Reference:
        Deep State-space Gaussian processes. 2020
    
    Args:
        t:  (n, ) Input (time)
    Return:
        y:  (n, ) y(t)
    """
    return np.sin(7 * np.pi * np.cos(2 * np.pi * (t ** 2))) ** 2 / \
           (np.cos(5 * np.pi * t) + 2)


def rect(t: np.ndarray) -> np.ndarray:
    """
    A magnitude-varing rectangle signal. Very challenging
    for a conventioanl GP to model.
    
    Reference:
        Deep State-space Gaussian processes. 2020
    
    Args:
        t:  (n, ) Input (time)
    Return:
        y:  (n, ) y(t)
    """
    # Scale to [0, 1]
    tau = (t - np.min(t)) / (np.max(t) - np.min(t))

    # Jumping points
    p = np.linspace(1 / 6, 5 / 6, 5)

    y = np.zeros(t.shape)
    y[(tau >= 0) & (tau < p[0])] = 0
    y[(tau >= p[0]) & (tau < p[1])] = 1
    y[(tau >= p[1]) & (tau < p[2])] = 0
    y[(tau >= p[2]) & (tau < p[3])] = 0.6
    y[(tau >= p[3]) & (tau < p[4])] = 0
    y[tau >= p[4]] = 0.4

    return y


def obs_noise(x: np.ndarray,
              r: float,
              seed: Optional[int] = None) -> np.ndarray:
    """
    Observe data x with Gaussian noises.
    y = x + r,   r ~ N(0, R)

    Parameters
    ----------
    x: np.ndarray (n, )
        The input
    r: float
        The noise variance
    seed

    Returns
    -------
    out: np.ndarray
        Noisified observations
    """
    rng = np.random.RandomState(seed)
    dtype = x.dtype
    return x + np.sqrt(r) * rng.normal(x, math.sqrt(r), (x.shape[0],)).astype(dtype)
