import math

import gpflow
import tensorflow as tf

from pssgp.kernels.base import ContinuousDiscreteModel, SDEKernelMixin
from pssgp.kernels.matern.common import get_matern_sde


class Matern52(gpflow.kernels.Matern52, SDEKernelMixin):
    __doc__ = gpflow.kernels.Matern52.__doc__

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = get_matern_sde(self.variance, self.lengthscales, 3)
        lengthscales = tf.reduce_sum(self.lengthscales)
        lamda = math.sqrt(5) / lengthscales
        variance = tf.reduce_sum(self.variance)

        temp = lamda ** 2 * variance
        P_infty = tf.linalg.diag([variance, temp / 3, temp ** 2])
        P_infty = tf.tensor_scatter_nd_sub(P_infty, [[0, 2], [2, 0]], [temp / 3, temp / 3])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
