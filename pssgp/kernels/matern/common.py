import math
from typing import Tuple

import gpflow.config as config
import numpy as np
import tensorflow as tf
from scipy.special import binom


def _get_transition_matrix(lamda: tf.Tensor, d: int, dtype: tf.DType) -> tf.Tensor:
    with tf.name_scope("get_transition_matrix"):
        F = tf.linalg.diag(tf.ones((d - 1,), dtype=dtype), k=1, num_cols=d, num_rows=d)
        binomial_coeffs = binom(d, np.arange(0, d, dtype=int)).astype(dtype)
        binomial_coeffs = tf.convert_to_tensor(binomial_coeffs, dtype=dtype)
        lambda_powers = lamda ** np.arange(d, 0, -1, dtype=dtype)
        update_indices = [[d - 1, k] for k in range(d)]
        F = tf.tensor_scatter_nd_sub(F, update_indices, lambda_powers * binomial_coeffs)
        return F


def _get_brownian_cov(variance, lamda, d, dtype) -> tf.Tensor:
    q = (2 * lamda) ** (2 * d - 1) * variance * math.factorial(d - 1) ** 2 / math.factorial(2 * d - 2)
    return q * tf.eye(1, dtype=dtype)


def get_matern_sde(variance, lengthscales, d: int) -> Tuple[tf.Tensor, ...]:
    """
    TODO: write description

    Parameters
    ----------
    variance
    lengthscales
    d: int
        the exponent of the Matern kernel plus one half
        for instance Matern32 -> 2, this will be used as the dimension of the latent SSM

    Returns
    -------
    F, L, H, Q: tuple of tf.Tensor
        Parameters for the LTI sde
    """
    dtype = config.default_float()
    lamda = math.sqrt(2 * d - 1) / lengthscales
    F = _get_transition_matrix(lamda, d, dtype)
    one = tf.ones((1,), dtype)
    L = tf.linalg.diag(one, k=-d + 1, num_rows=d, num_cols=1)  # type: tf.Tensor
    H = tf.linalg.diag(one, num_rows=1, num_cols=d)  # type: tf.Tensor
    Q = _get_brownian_cov(variance, lamda, d, dtype)
    return F, L, H, Q
