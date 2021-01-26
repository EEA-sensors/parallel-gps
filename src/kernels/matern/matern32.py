import math

import gpflow
import tensorflow as tf

from src.kernels.base import ContinuousDiscreteModel, SDEKernelMixin
from src.kernels.matern.common import get_matern_sde


class Matern32(gpflow.kernels.Matern32, SDEKernelMixin):
    __doc__ = gpflow.kernels.Matern32.__doc__

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = get_matern_sde(self.variance, self.lengthscales, 2)

        lengthscales = tf.reduce_sum(self.lengthscales)
        lamda = math.sqrt(3) / lengthscales
        variance = tf.reduce_sum(self.variance)

        P_infty = tf.linalg.diag(tf.stack([variance, lamda ** 2 * variance], axis=0))
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
