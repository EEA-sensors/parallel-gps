import gpflow
import tensorflow as tf

from pssgp.kernels.base import ContinuousDiscreteModel, SDEKernelMixin, get_lssm_spec
from pssgp.kernels.matern.common import get_matern_sde


class Matern12(SDEKernelMixin, gpflow.kernels.Matern12):
    __doc__ = gpflow.kernels.Matern12.__doc__

    def __init__(self, variance=1.0, lengthscales=1.0, **kwargs):
        gpflow.kernels.Matern12.__init__(self, variance, lengthscales, **kwargs)
        SDEKernelMixin.__init__(self, **kwargs)

    def get_spec(self, T):
        return get_lssm_spec(1, T)

    def get_sde(self) -> ContinuousDiscreteModel:
        F, L, H, Q = get_matern_sde(self.variance, self.lengthscales, 1)
        variance = tf.reduce_sum(self.variance)

        P_infty = tf.linalg.diag([variance])
        return ContinuousDiscreteModel(P_infty, F, L, H, Q)
