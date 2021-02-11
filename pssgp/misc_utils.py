import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Union

ACC_TYPES = Union[tf.Tensor, np.ndarray]


def rmse(x1: ACC_TYPES, x2: ACC_TYPES) -> tf.Tensor:
    """Root mean square error
    """
    x1 = tf.reshape(x1, (-1, ))
    x2 = tf.reshape(x2, (-1, ))
    return tf.math.sqrt(tf.reduce_mean(tf.math.square(x1 - x2)))


def error_shade(t: ACC_TYPES, m: ACC_TYPES, cov: ACC_TYPES, **kwargs):
    """Plot .95 confidence interval
    """
    t = tf.reshape(t, (-1, ))
    m = tf.reshape(m, (-1,))
    cov = tf.reshape(cov, (-1,))
    plt.fill_between(t,
                     m - 1.96 * np.sqrt(cov),
                     m + 1.96 * np.sqrt(cov),
                     **kwargs)