import gpflow
import tensorflow as tf

from src.kernels.base import ContinuousDiscreteModel, SDEKernelMixin
from src.kernels.matern.common import get_matern_sde


class Matern12(gpflow.kernels.Matern12, SDEKernelMixin):
    __doc__ = gpflow.kernels.Matern12.__doc__
    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, Q, H = get_matern_sde(self.variance, self.lengthscales, 1)
        P_infty = tf.linalg.diag([self.variance])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
