import math

import gpflow
import tensorflow as tf

from src.kernels.base import ContinuousDiscreteModel, SDEKernelMixin
from src.kernels.matern.common import get_matern_sde


class Matern52(SDEKernelMixin, gpflow.kernels.Matern52):
    __doc__ = gpflow.kernels.Matern52.__doc__

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, Q, H = get_matern_sde(self.variance, self.lengthscales, 3)
        lamda = math.sqrt(5) / self.lengthscales
        temp = lamda ** 2 * self.variance
        P_infty = tf.linalg.diag([self.variance, temp / 3, temp ** 2])
        P_infty = tf.tensor_scatter_nd_sub(P_infty, [[0, 2], [2, 0]], [temp / 3, temp / 3])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
