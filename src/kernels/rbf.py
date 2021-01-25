import math
from typing import Tuple

import gpflow
import tensorflow as tf

from src.kernels.base import ContinuousDiscreteModel, SDEKernelMixin

class RBF(gpflow.kernels.RBF, SDEKernelMixin):
    __doc__ = gpflow.kernels.RBF.__doc__

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = get_matern_sde(self.variance, self.lengthscales, 3)
        lengthscales = tf.reduce_sum(self.lengthscales)
        lamda = math.sqrt(5) / lengthscales
        variance = tf.reduce_sum(self.variance)

        temp = lamda ** 2 * variance
        P_infty = tf.linalg.diag([variance, temp / 3, temp ** 2])
        P_infty = tf.tensor_scatter_nd_sub(P_infty, [[0, 2], [2, 0]], [temp / 3, temp / 3])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)

    def get_rbf_sde(variance,
                    lengthscales,
                    order) -> Tuple[tf.Tensor, ...]:
        pass