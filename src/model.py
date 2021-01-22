import tensorflow as tf
from gpflow import Parameter
from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive

from src.kalman.parallel import pkf, pkfs
from src.kalman.sequential import kf, kfs


@tf.function
def _merge_sorted(a, b, *args):
    """
    Merge sorted arrays efficiently, inspired by https://stackoverflow.com/a/54131815

    Parameters
    ----------
    a: tf.Tensor
        Sorted tensor for ordering
    b: tf.Tensor
        Sorted tensor for ordering
    args: list of tuple of tf.Tensor
        Some data ordered according to a and b that need to be merged whilst keeping the order.

    Returns
    -------
    cs: list of tf.Tensor
        Merging of a_x and b_x in the right order.

    """

    assert len(a.shape) == len(b.shape) == 1
    a_shape, b_shape = a.shape[0], b.shape[0]
    c_len = a_shape + b_shape
    if a_shape < b_shape:
        a, b = b, a
        args = tuple((j, i) for i, j in args)
        a_shape, b_shape = a.shape[0], b.shape[0]
    b_indices = tf.range(b_shape, dtype=tf.int32) + tf.searchsorted(a, b)
    a_indices = tf.ones((c_len,), dtype=tf.bool)
    a_indices = tf.tensor_scatter_nd_update(a_indices, b_indices[:, None], tf.zeros_like(b_indices, tf.bool))
    c_range = tf.range(c_len, dtype=tf.int32)
    a_mask = tf.boolean_mask(c_range, a_indices)[:, None]

    def _inner_merge(u, v):
        c = tf.zeros((c_len,) + u.shape[1:], dtype=u.dtype)
        c = tf.tensor_scatter_nd_update(c, b_indices[:, None], v)
        c = tf.tensor_scatter_nd_update(c,
                                        a_mask,
                                        u)
        return c

    return (_inner_merge(a, b),) + tuple(_inner_merge(i, j) for i, j in args)


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
        squeezed_ts = tf.squeeze(ts)
        squeezed_Xnew = tf.squeeze(Xnew)
        float_ys = float("nan") * tf.ones((Xnew.shape[0], ys.shape[1]), dtype=ys.dtype)
        all_ts, all_ys, all_flags = _merge_sorted(squeezed_ts, squeezed_Xnew,
                                                  (ys, float_ys),
                                                  (tf.zeros_like(squeezed_ts, dtype=tf.bool),
                                                   tf.ones_like(squeezed_Xnew, dtype=tf.bool)))
        #  this merging is equivalent to using argsort but uses O(log(T)^2) operations instead.

        ssm = self._make_model(all_ts[:, None])
        sms, sPs = self._kfs(ssm, all_ys)
        return tf.boolean_mask(sms, all_flags, 0), tf.boolean_mask(sPs, all_flags, 0)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        ts, Y = self.data
        ssm = self._make_model(ts)
        *_, ll = self._kf(ssm, Y, True)
        return ll
