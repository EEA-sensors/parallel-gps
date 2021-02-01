from functools import partial
from typing import Tuple

import math
import gpflow
import gpflow.config as config
import numba as nb
import numpy as np
import tensorflow as tf

from scipy.special import factorial, comb
from gpflow.kernels import SquaredExponential
from pssgp.kernels.base import ContinuousDiscreteModel, SDEKernelMixin

tf_kron = tf.linalg.LinearOperatorKronecker


def _get_offline_coeffs(N) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get coefficients which are independent of parameters (ell, sigma, and period). That are, fixed.

    Parameters
    ----------
    N : Approximation order of periodic state-space model.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray

    See Also
    --------
    seriescoeff.m

    """
    r = np.arange(0, N + 1)
    J, K = np.meshgrid(r, r)
    div_facto_K = factorial(K)

    # Get b(K, J)
    b = 2 * comb(K, np.floor((K - J) / 2) * (J <= K), exact=True) / \
        (1 + (J == 0) * 1) * (J <= K) * (np.mod(K - J, 2) == 0)
    return b, K, div_facto_K


class Periodic(gpflow.kernels.periodic, SDEKernelMixin):
    __doc__ = gpflow.kernels.periodic.__doc__

    def __init__(self, variance=1.0, lengthscales=1.0, period=1., **kwargs):
        """
        Note: The gpflow has a different periodic covariance function.

        Periodic in gpflow is: k(r) =  σ² exp{ -0.5 sin²(π r / γ) / ℓ²},
        ours: k(r) =  σ² exp{ -2 sin²(w r / 2) / ℓ²},  where w = 2 π / γ
        To keep consistence, we need to scale down self.lengthscales by 4, i.e., self.lengthscales / 4
        """
        self._order = kwargs.pop('order', 6)
        base_cov = SquaredExponential(variance, lengthscales)
        super().__init__(base_cov, period, **kwargs)

    def get_sde(self) -> ContinuousDiscreteModel:
        dtype = config.default_float()
        N = self._order
        w0 = 2 * math.pi / self.period

        # Prepare offline fixed coefficients
        b, K, div_facto_K = _get_offline_coeffs(N)
        b = tf.constant(b, dtype=dtype)
        K = tf.constant(K, dtype=dtype)
        div_facto_K = tf.constant(div_facto_K, dtype=dtype)

        op_F = tf.linalg.LinearOperatorFullMatrix(tf.constant([[0, -w0], [w0, 0]], dtype=dtype))
        op_diag = tf.linalg.LinearOperatorDiag(tf.range(0, N + 1, dtype=dtype))
        F = tf_kron(op_diag, op_F).to_dense()

        L = tf.eye(2 * (N + 1), dtype=dtype)

        Q = tf.zeros((2 * (N + 1), 2 * (N + 1)), dtype=dtype)

        q2 = b * self.lengthscales ** (-2 * K) * div_facto_K * tf.math.exp(self.lengthscales ** (-2)) * \
             2 ** (-K) * self.variances

        Pinf = tf_kron(q2, tf.linalg.LinearOperatorIdentity(2, dtype=dtype))

        H = tf_kron(tf.linalg.LinearOperatorFullMatrix(tf.ones((1, N + 1), dtype=dtype)),
                    tf.linalg.LinearOperatorFullMatrix(tf.constant([1, 0], shape=(1, 2), dtype=dtype))).to_dense()

        return ContinuousDiscreteModel(Pinf, F, L, H, Q)
