"""
Draw a sample from a GP.

"""

import numpy as np
from scipy.linalg import expm


# from ..kalman import StateSpaceModel

def draw_gp_batch(gp: None,
                  T: np.ndarray) -> np.ndarray:
    """
    Draw a sample from an GP given cov and mean functions.
    
    Args:
        gp: GP object
        T:  time instances (t1, t2, ...)
        
    Returns:
        f(t1, t2, ...)
    """
    m = 0
    cov = 0

    return m + np.linalg.cholesky(cov) @ np.random.randn(T.shape[0])


def draw_gp_ss(gp: None,
               T: np.ndarray,
               t0: float,
               m0: np.ndarray,
               P0: np.ndarray) -> np.ndarray:
    """
    Draw a GP sample from a LTI SDE.
    
    dx = F x dt + L dW,   E[dWdW] = \dirac
    
    x(t) = exp(F (t - t0)) x(t0) + \int^t_{t0} exp(F (t - s)) L dW(s)
    
    Note that it must be LTI, or the matrix exponentional breaks.
    """
    # Get F, L, Q etc from StateSpaceModel
    F = 0
    L = 0

    # Draw a init point
    x0 = m0 + np.linalg.cholesky(P0) @ np.random.rand(m0.shape[0])

    # Draw dW
    T_int = np.interp()
    dW = np.random.randn(T_int.shape[0])

    # Solution
    x = np.zeros(T.shape)
    ito_itegrand = np.zeros(T.shape)

    # ODE and Riemannian
    for id, t in enumerate(T):

        x[id] = expm(F * (t - t0)) @ x0

        # Riemannian
        for id2, s in [t0, t]:
            ito_itegrand[idx] = expm(F * (t - s)) * L * dW[0]

        x[id] += x[id] + np.sum(ito_itegrand)

    return x
