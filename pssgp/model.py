from functools import partial

import tensorflow as tf
from gpflow import Parameter, config
from gpflow.models import GPModel
from gpflow.models.model import MeanAndVariance
from gpflow.models.training_mixins import InputData, RegressionData, InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor
from gpflow.utilities import positive

from pssgp.kalman.parallel import pkf, pkfs
from pssgp.kalman.sequential import kf, kfs


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
    with tf.name_scope("merge_sorted"):
        assert len(a.shape) == len(b.shape) == 1
        a_shape, b_shape = tf.shape(a)[0], tf.shape(b)[0]
        c_len = tf.shape(a)[0] + tf.shape(b)[0]
        if a_shape < b_shape:
            a, b = b, a
            a_shape, b_shape = tf.shape(a)[0], tf.shape(b)[0]
            args = tuple((j, i) for i, j in args)
        b_indices = tf.range(b_shape, dtype=tf.int32) + tf.searchsorted(a, b)
        a_indices = tf.ones((c_len,), dtype=tf.bool)
        a_indices = tf.tensor_scatter_nd_update(a_indices, b_indices[:, None], tf.zeros_like(b_indices, tf.bool))
        c_range = tf.range(c_len, dtype=tf.int32)
        a_mask = tf.boolean_mask(c_range, a_indices)[:, None]

        def _inner_merge(u, v):
            c = tf.concat([u, v], 0)
            c = tf.tensor_scatter_nd_update(c, b_indices[:, None], v)
            c = tf.tensor_scatter_nd_update(c, a_mask, u)
            return c

        return (_inner_merge(a, b),) + tuple(_inner_merge(i, j) for i, j in args)


class StateSpaceGP(GPModel, InternalDataTrainingLossMixin):
    def __init__(self,
                 data: RegressionData,
                 kernel,
                 noise_variance: float = 1.0,
                 parallel=False,
                 max_parallel=10000
                 ):

        self.noise_variance = Parameter(noise_variance, transform=positive())
        ts, ys = data_input_to_tensor(data)
        super().__init__(kernel, None, None, num_latent_gps=ys.shape[-1])
        self.data = ts, ys
        filter_spec = kernel.get_spec(ts.shape[0])
        filter_ys_spec = tf.TensorSpec((ts.shape[0], 1), config.default_float())
        smoother_spec = kernel.get_spec(None)
        smoother_ys_spec = tf.TensorSpec((None, 1), config.default_float())

        if not parallel:
            self._kf = tf.function(partial(kf, return_loglikelihood=True, return_predicted=False),
                                   input_signature=[filter_spec, filter_ys_spec])
            self._kfs = tf.function(kfs, input_signature=[smoother_spec, smoother_ys_spec])
        else:
            self._kf = tf.function(partial(pkf, return_loglikelihood=True, max_parallel=ts.shape[0]),
                                   input_signature=[filter_spec, filter_ys_spec])
            self._kfs = tf.function(partial(pkfs, max_parallel=max_parallel),
                                    input_signature=[smoother_spec, smoother_ys_spec])

    def _make_model(self, ts):
        with tf.name_scope("make_model"):
            R = tf.reshape(self.noise_variance, (1, 1))
            ssm = self.kernel.get_ssm(ts, R)
        return ssm

    def predict_f(
            self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        ts, ys = self.data
        Xnew = tf.convert_to_tensor(Xnew, dtype=config.default_float())
        squeezed_ts = tf.squeeze(ts)
        squeezed_Xnew = tf.squeeze(Xnew)
        float_ys = float("nan") * tf.ones((Xnew.shape[0], ys.shape[1]), dtype=ys.dtype)
        all_ts, all_ys, all_flags = _merge_sorted(squeezed_ts, squeezed_Xnew,
                                                  (ys, float_ys),
                                                  (tf.zeros_like(squeezed_ts, dtype=tf.bool),
                                                   tf.ones_like(squeezed_Xnew, dtype=tf.bool)))
        #  this merging is equivalent to using argsort but uses O(log(T)) operations instead.
        ssm = self._make_model(all_ts[:, None])
        sms, sPs = self._kfs(ssm, all_ys)
        res = tf.boolean_mask(sms, all_flags, 0), tf.boolean_mask(sPs, all_flags, 0)
        return tf.linalg.matvec(ssm.H, res[0]), tf.linalg.diag_part(tf.linalg.matmul(ssm.H,
                                                                                     tf.linalg.matmul(res[1],
                                                                                                      ssm.H,
                                                                                                      transpose_b=True)))

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        ts, Y = self.data
        ssm = self._make_model(ts)
        fms, fPs, ll = self._kf(ssm, Y)
        return ll
