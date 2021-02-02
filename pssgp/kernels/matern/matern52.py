import math

import gpflow
import tensorflow as tf

from pssgp.kernels.base import ContinuousDiscreteModel, SDEKernelMixin, get_lssm_spec
from pssgp.kernels.matern.common import get_matern_sde


class Matern52(SDEKernelMixin, gpflow.kernels.Matern52):
    __doc__ = gpflow.kernels.Matern52.__doc__

    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        self._order = kwargs.pop('order', 3)
        gpflow.kernels.Matern52.__init__(self, variance, lengthscales, **kwargs)
        SDEKernelMixin.__init__(self, **kwargs)

    def get_spec(self, T):
        return get_lssm_spec(3, T)

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = get_matern_sde(self.variance, self.lengthscales, 3)
        lengthscales = tf.reduce_sum(self.lengthscales)
        lamda = math.sqrt(5) / lengthscales
        variance = tf.reduce_sum(self.variance)

        temp = lamda ** 2 * variance
        P_infty = tf.linalg.diag([variance, temp / 3, temp ** 2])
        P_infty = tf.tensor_scatter_nd_sub(P_infty, [[0, 2], [2, 0]], [temp / 3, temp / 3])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
