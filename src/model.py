import tensorflow as tf
from gpflow import Parameter
from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive

from src.kalman.parallel import pkf, pkfs
from src.kalman.sequential import kf, kfs


class StateSpaceGP(GPModel):
    def __init__(self,
                 data: RegressionData,
                 kernel,
                 noise_variance: float = 1.0,
                 parallel=False
                 ):
        self._noise_variance = Parameter(noise_variance, transform=positive())
        ts, ys = data_input_to_tensor(data)
        super().__init__(kernel, None, None, num_latent_gps=ys.shape[-1])
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
        flags = tf.concat([tf.cast(tf.zeros_like(ts), tf.bool),
                           tf.cast(tf.ones_like(Xnew), tf.bool)], axis=0)
        all_ts = tf.concat([ts, Xnew], axis=0)
        all_ts_argsort = tf.argsort(all_ts, 0)
        all_ts = tf.gather(all_ts, all_ts_argsort, axis=0)
        all_ts = tf.reshape(all_ts, (all_ts.shape[0], 1))

        all_ys = tf.concat([ys,
                            float("nan") * tf.ones((Xnew.shape[0], ys.shape[1]), dtype=ys.dtype)],
                           axis=0)
        sorted_flags = tf.squeeze(tf.gather(flags, all_ts_argsort))
        ssm = self._make_model(all_ts)
        sms, sPs = self._kf(ssm, all_ys)
        return tf.boolean_mask(sms, sorted_flags, 0), tf.boolean_mask(sPs, sorted_flags, 0)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        ts, Y = self.data
        ssm = self._make_model(ts)
        *_, ll = self._kf(ssm, Y, True)
        return ll
