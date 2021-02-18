from functools import partial
from typing import Tuple, Union, List

import math
import gpflow
import gpflow.config as config
import numba as nb
import numpy as np
import tensorflow as tf

from scipy.special import factorial, comb
from gpflow.kernels import SquaredExponential
from pssgp.kernels.base import ContinuousDiscreteModel, SDEKernelMixin, get_lssm_spec

tf_kron = tf.linalg.LinearOperatorKronecker


def _get_offline_coeffs(N) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get coefficients which are independent of parameters (ell, sigma, and period). That are, fixed.

    Parameters
    ----------
    N : Approximation order of periodic state-space model.

    Returns
    -------
    b: np.ndarray
    K: np.ndarray
    div_facto_K: np.ndarray

    See Also
    --------
    seriescoeff.m

    """
    r = np.arange(0, N + 1)
    J, K = np.meshgrid(r, r)
    div_facto_K = 1 / factorial(K)
    # Get b(K, J)
    b = 2 * comb(K, np.floor((K - J) / 2) * (J <= K)) / \
        (1 + (J == 0)) * (J <= K) * (np.mod(K - J, 2) == 0)
    return b, K, div_facto_K


class Periodic(SDEKernelMixin, gpflow.kernels.Periodic):
    __doc__ = gpflow.kernels.Periodic.__doc__

    def __init__(self, base_kernel: SquaredExponential, period: Union[float, List[float]] = 1.0, **kwargs):
        assert isinstance(base_kernel, SquaredExponential), "Only SquaredExponential is supported at the moment"
        self._order = kwargs.pop('order', 6)
        gpflow.kernels.Periodic.__init__(self, base_kernel, period)
        SDEKernelMixin.__init__(self, **kwargs)

    def get_spec(self, T):
        return get_lssm_spec(2 * (self._order + 1), T)

    def get_sde(self) -> ContinuousDiscreteModel:
        dtype = config.default_float()
        N = self._order
        w0 = 2 * math.pi / self.period
        lengthscales = self.base_kernel.lengthscales * 2.

        # Prepare offline fixed coefficients
        b, K, div_facto_K = _get_offline_coeffs(N)
        b = tf.convert_to_tensor(b, dtype=dtype)
        K = tf.convert_to_tensor(K, dtype=dtype)
        div_facto_K = tf.convert_to_tensor(div_facto_K, dtype=dtype)

        op_F = tf.linalg.LinearOperatorFullMatrix([[0, -w0], [w0, 0]])
        op_diag = tf.linalg.LinearOperatorDiag(np.arange(0, N + 1, dtype=dtype))
        F = tf_kron([op_diag, op_F]).to_dense()

        L = tf.eye(2 * (N + 1), dtype=dtype)

        Q = tf.zeros((2 * (N + 1), 2 * (N + 1)), dtype=dtype)

        q2 = b * lengthscales ** (-2 * K) * div_facto_K * tf.math.exp(-lengthscales ** (-2)) * \
             2 ** (-K) * self.base_kernel.variance
        q2 = tf.linalg.LinearOperatorDiag(tf.reduce_sum(q2, axis=0))

        Pinf = tf_kron([q2, tf.linalg.LinearOperatorIdentity(2, dtype=dtype)]).to_dense()

        H = tf_kron([tf.linalg.LinearOperatorFullMatrix(tf.ones((1, N + 1), dtype=dtype)),
                     tf.linalg.LinearOperatorFullMatrix(tf.constant([[1, 0]], dtype=dtype))]).to_dense()
        return ContinuousDiscreteModel(Pinf, F, L, H, Q)
