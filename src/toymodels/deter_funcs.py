"""
Some toy models based on deterministic test functions
"""
import numpy as np

def sinu():
    pass

def rect():
    pass

def obs_noise(x: np.ndarray, 
              H: np.ndarray, 
              R: np.ndarray) -> np.ndarray:
    """
    Observe data x linearly with Gaussian noises. 
    y = H x + r,   r ~ N(0, R)
    """
    
    return H @ x + np.linalg.cholesky(R) @ np.random.randn(x.shape)