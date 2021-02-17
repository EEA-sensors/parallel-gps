from functools import partial
from typing import Tuple, Optional

import numba as nb
import numpy as np
import tensorflow as tf
from gpflow import config


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


def balance_ss(F: tf.Tensor,
               L: tf.Tensor,
               H: tf.Tensor,
               q: tf.Tensor,
               n_iter: int = 5) -> Tuple[tf.Tensor, ...]:
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
    P: tf.Tensor, optional
        ...
    n_iter : int
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
    d = tf.numpy_function(partial(nb_balance_ss, iter=n_iter), (F,), dtype)
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


def solve_lyap_vec(F: tf.Tensor,
                   L: tf.Tensor,
                   Q: tf.Tensor) -> tf.Tensor:
    """Vectorized Lyapunov equation solver

    F P + P F' + L Q L' = 0

    Parameters
    ----------
    F : tf.Tensor
        ...
    L : tf.Tensor
        ...
    Q : tf.Tensor
        ...

    Returns
    -------
    Pinf : tf.Tensor
        Steady state covariance

    """
    dtype = config.default_float()

    dim = tf.shape(F)[0]

    op1 = tf.linalg.LinearOperatorFullMatrix(F)
    op2 = tf.linalg.LinearOperatorIdentity(dim, dtype=dtype)

    F1 = tf.linalg.LinearOperatorKronecker([op2, op1]).to_dense()
    F2 = tf.linalg.LinearOperatorKronecker([op1, op2]).to_dense()


    F = F1 + F2
    Q = tf.matmul(L, tf.matmul(Q, L, transpose_b=True))
    Pinf = tf.reshape(tf.linalg.solve(F, tf.reshape(Q, (-1, 1))), (dim, dim))
    Pinf = -0.5 * (Pinf + tf.transpose(Pinf))
    return Pinf