import gpflow
from gpflow import Parameter, config
from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.util import data_input_to_tensor
import tensorflow as tf

from src.kalman.parallel import pkf, pks, pkfs
from src.kalman.sequential import kf, ks, kfs
from src.kernels.base import SDEKernelMixin


class StateSpaceGP(GPModel):
    def __init__(self,
                 data: RegressionData,
                 kernel,
                 noise_variance: float = 1.0,
                 parallel=False
                 ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        self._noise_variance = Parameter(noise_variance)
        ts, ys = data_input_to_tensor(data)
        super().__init__(kernel, likelihood, None, num_latent_gps=ys.shape[-1])
        self.data = ts, ys
        if not parallel:
            self._kf = kf
            self._kfs = kfs
        else:
            self._kf = pkf
            self._kfs = pkfs

    def _make_model(self, ts):
        R = tf.reshape(self._noise_variance, (1, 1))
        ssm = self.kernel.get_ssm(ts, R)
        return ssm

    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        ts, ys = self.data
        flags = tf.concat([tf.zeros_like(ts).astype(tf.bool),
                           tf.ones_like(Xnew).astype(tf.bool)], axis=0)
        all_ts = tf.concat([ts, Xnew], axis=0)
        all_ts_argsort = tf.argsort(all_ts)
        all_ts = all_ts[all_ts_argsort]
        all_ys = tf.concat([ys,
                            float("nan") * tf.ones((Xnew.shape[0], ys.shape[1]), dtype=ys.dtype, device=ys.device)],
                           axis=0)
        sorted_flags = flags[all_ts_argsort]
        ssm = self._make_model(all_ts)
        sms, sPs = self._kfs(ssm, all_ys)
        return MeanAndVariance(tf.boolean_mask(sms, sorted_flags, 0),
                               tf.boolean_mask(sPs, sorted_flags, 0))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        ts, Y = self.data
        ssm = self._make_model(ts)
        *_, ll = self._kf(ssm, Y, True)
        return ll


