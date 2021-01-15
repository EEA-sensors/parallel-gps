from typing import Tuple

import tensorflow as tf

from src.kernels.base import ContinuousDiscreteModel


def sde_to_lgssm(sde: ContinuousDiscreteModel, dt: tf.Tensor) -> Tuple[tf.Tensor, ...]:
    dtype = sde.F.dtype
    A = tf.linalg.expm(sde.F * dt)
    n = sde.F.shape[0]
    Phi_high = tf.stack([sde.F,
                         sde.L @ tf.linalg.matmul(sde.Q, sde.L, transpose_b=True)],
                        axis=1)
    Phi_low = tf.stack([tf.zeros((n, n), dtype=dtype), -tf.transpose(F)], axis=1)
    Phi = tf.stack([Phi_high, Phi_low], axis=0)
    AB = tf.linalg.expm(Phi * dt) @ tf.stack([tf.zeros((n, n), dtype=dtype), tf.eye(n, dtype)], axis=0)
    Q = tf.linalg.matmul(AB[:n], A, transpose_b=True)
    return A, Q
