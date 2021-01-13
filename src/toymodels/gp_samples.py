"""
Draw a sample from a GP.

To tackle with many measurements, we draw samples from SDEs by
Euler--Maruyama
"""

import numpy as np

from ..kalman import StateSpaceModel

def draw_gp(gp: StateSpaceModel, 
            T: np.ndarray, 
            dt: float, 
            N: int, 
            int_steps: int, 
            method: str = 'EM') -> np.ndarray:
    """
    Draw a sample from an GP.
    
    Args:
        gp:         GP object
        dt:         Time interval
        N:          Number of measurements
        int_steps:  Integration steps
        method:     Simulation method
    """
    pass