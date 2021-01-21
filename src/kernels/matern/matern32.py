import math

import gpflow
import tensorflow as tf

from src.kernels.base import ContinuousDiscreteModel, SDEKernelMixin
from src.kernels.matern.common import get_matern_sde


class Matern32(gpflow.kernels.Matern32, SDEKernelMixin):
    __doc__ = gpflow.kernels.Matern32.__doc__

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, Q, H = get_matern_sde(self.variance, self.lengthscales, 2)
        lamda = math.sqrt(3) / self.lengthscales
        P_infty = tf.linalg.diag([self.variance, lamda ** 2 * self.variance])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
