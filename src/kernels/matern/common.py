import math
from typing import Tuple

import gpflow.config as config
import numpy as np
import tensorflow as tf
from scipy.special import binom


def _get_transition_matrix(lamda, d, dtype) -> tf.Tensor:
    F = tf.linalg.diag(tf.ones((d - 1,), dtype=dtype), k=1, num_cols=d, num_rows=d)
    binomial_coeffs = binom(d, np.arange(0, d, dtype=int)).astype(dtype)
    binomial_coeffs = tf.convert_to_tensor(binomial_coeffs)
    lambda_powers = lamda ** np.arange(d, 0, -1, dtype=dtype)
    update_indices = [[d - 1, k] for k in range(d)]
    F = tf.tensor_scatter_nd_sub(F, update_indices, lambda_powers * binomial_coeffs)
    return F


def _get_brownian_cov(variance, lengthscales, d, dtype) -> tf.Tensor:
    q = (2 * lengthscales) ** (2 * d - 1) * variance * math.factorial(d - 1) ** 2 / math.factorial(2 * d - 2)
    return q * tf.eye(1, dtype=dtype)


def get_matern_sde(variance, lengthscales, d) -> Tuple[tf.Tensor, ...]:
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
    L = tf.linalg.diag([1.], k=-d, num_rows=d, num_cols=1)  # type: tf.Tensor
    H = tf.linalg.diag([1.], num_rows=1, num_cols=d)  # type: tf.Tensor
    Q = _get_brownian_cov(variance, lengthscales, d, dtype)
    return F, L, H, Q