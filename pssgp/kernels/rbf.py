import math
from functools import partial
from typing import Tuple

import gpflow
import gpflow.config as config
import numba as nb
import numpy as np
import tensorflow as tf

from pssgp.kernels.base import ContinuousDiscreteModel, SDEKernelMixin, get_lssm_spec
from pssgp.math_utils import solve_lyap_vec


def _get_unscaled_rbf_sde(order: int = 6) -> Tuple[np.ndarray, ...]:
    """Get un-scaled RBF SDE.
    Pre-computed before loading to tensorflow.

    Parameters
    ----------
    order : int, default=6
        Order of Taylor expansion

    Returns
    -------
    F, L, H, Q : np.ndarray
        SDE coefficients.

    See Also
    --------
    se_to_ss.m
    """
    dtype = config.default_float()
    B = math.sqrt(2 * math.pi)
    A = np.zeros((2 * order + 1,), dtype=dtype)

    i = 0
    for k in range(order, -1, -1):
        A[i] = 0.5 ** k / math.factorial(k)
        i = i + 2

    q = B / np.polyval(A, 0)

    LA = np.real(A / (1j ** np.arange(A.size - 1, -1, -1, dtype=dtype)))

    AR = np.roots(LA)

    GB = 1
    GA = np.poly(AR[np.real(AR) < 0])

    GA = GA / GA[-1]

    GB = GB / GA[0]
    GA = GA / GA[0]

    F = np.zeros((GA.size - 1, GA.size - 1), dtype=dtype)
    F[-1, :] = -GA[:0:-1]
    F[:-1, 1:] = np.eye(GA.size - 2, dtype=dtype)

    L = np.zeros((GA.size - 1, 1), dtype=dtype)
    L[-1, 0] = 1

    H = np.zeros((1, GA.size - 1), dtype=dtype)
    H[0, 0] = GB

    return F, L, H, q


@partial(nb.jit, nopython=True)
def nb_balance_ss(F: np.ndarray,
                  iter: int) -> np.ndarray:
    dim = F.shape[0]
    dtype = F.dtype
    d = np.ones((dim,), dtype=dtype)
    for k in range(iter):
        for i in range(dim):
            tmp = np.copy(F[:, i])
            tmp[i] = 0.
            c = np.linalg.norm(tmp, 2)
            tmp2 = np.copy(F[i, :])
            tmp2[i] = 0.

            r = np.linalg.norm(tmp2, 2)
            f = np.sqrt(r / c)
            d[i] *= f
            F[:, i] *= f
            F[i, :] /= f
    return d


def _balance_ss(F: tf.Tensor,
                L: tf.Tensor,
                H: tf.Tensor,
                q: tf.Tensor,
                iter: int = 5) -> Tuple[tf.Tensor, ...]:
    """Balance state-space model to have better numerical stability

    Parameters
    ----------
    F : tf.Tensor
        Matrix
    L : tf.Tensor
        Matrix
    H : tf.Tensor
        Measurement matrix
    q : tf.Tensor
        Spectral dnesity
    iter : int
        Iteration of balancing

    Returns
    -------
    F : tf.Tensor
        ...
    L : tf.Tensor
        ...
    H : tf.Tensor
        ...
    q : tf.Tensor
        ...


    References
    ----------
    https://arxiv.org/pdf/1401.5766.pdf
    """
    dtype = config.default_float()
    d = tf.numpy_function(partial(nb_balance_ss, iter=iter), (F,), dtype)
    d = tf.reshape(d, (tf.shape(F)[0],))  # This is to make sure that the shape of d is known at compilation time.
    F = F * d[None, :] / d[:, None]
    L = L / d[:, None]
    H = H * d[None, :]

    tmp3 = tf.reduce_max(tf.abs(L))
    L = L / tmp3
    q = (tmp3 ** 2) * q

    tmp4 = tf.reduce_max(tf.abs(H))
    H = H / tmp4
    q = (tmp4 ** 2) * q

    return F, L, H, q


class RBF(SDEKernelMixin, gpflow.kernels.RBF):
    __doc__ = gpflow.kernels.RBF.__doc__

    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        self._order = kwargs.pop('order', 3)
        self._balancing_iter = kwargs.pop('balancing_iter', 5)
        gpflow.kernels.RBF.__init__(self, variance, lengthscales)
        SDEKernelMixin.__init__(self, **kwargs)

    __init__.__doc__ = r"""TODO: talk about order params \n\n""" + gpflow.kernels.RBF.__init__.__doc__

    def get_spec(self, T):
        return get_lssm_spec(self._order, T)

    def get_sde(self) -> ContinuousDiscreteModel:
        F_, L_, H_, q_ = _get_unscaled_rbf_sde(self._order)

        dtype = config.default_float()
        F = tf.convert_to_tensor(F_, dtype=dtype)
        L = tf.convert_to_tensor(L_, dtype=dtype)
        H = tf.convert_to_tensor(H_, dtype=dtype)
        q = tf.convert_to_tensor(q_, dtype=dtype)

        dim = F.shape[0]

        ell_vec = self.lengthscales ** tf.range(dim, 0, -1, dtype=dtype)
        update_indices = [[dim - 1, k] for k in range(dim)]
        F = tf.tensor_scatter_nd_update(F, update_indices, F[-1, :] / ell_vec)

        H = H / (self.lengthscales ** dim)
        q = self.variance * self.lengthscales * q

        Fb, Lb, Hb, qb = _balance_ss(F, L, H, q, self._balancing_iter)

        Pinf = solve_lyap_vec(Fb, Lb, qb)

        Q = tf.reshape(qb, (1, 1))
        return ContinuousDiscreteModel(Pinf, Fb, Lb, Hb, Q)
