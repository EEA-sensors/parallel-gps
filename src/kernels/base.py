import abc
from collections import namedtuple

import tensorflow as tf

from src.kalman.base import LGSSM

ContinuousDiscreteModel = namedtuple("ContinuousDiscreteModel", ["P0", "F", "L", "H", "Q"])


class SDEKernelMixin(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_sde(self) -> ContinuousDiscreteModel:
        """
        Creates the linear time invariant continuous discrete system associated to the stationary kernel at hand

        Returns
        -------
        sde: ContinuousDiscreteModel
            The associated LTI model
        """

    def get_ssm(self, ts, R):
        """
        Creates the linear Gaussian state space model associated to the stationary kernel at hand

        Parameters
        ----------
        ts: tf.Tensor
            The times at which we have observations
        R: tf.Tensor
            The observation covariance

        Returns
        -------
        lgssm: ContinuousDiscreteModel
            The associated state space model
        """
        sde = self.get_sde()
        n = sde.F.shape[0]
        dts = ts[1:] - ts[:-1]
        Fs = tf.linalg.expm(dts.reshape(-1, 1, 1) * tf.expand_dims(sde.F, 0))
        zeros = tf.zeros_like(sde.F)

        Phi = tf.stack(
            tf.stack([sde.F, sde.L @ tf.matmul(sde.Q, sde.L, transpose_b=True)], axis=1),
            tf.stack([zeros, -sde.F.T], axis=1),
            axis=0)

        AB = tf.linalg.expm(dts.reshape(-1, 1, 1) * tf.expand_dims(Phi, 0))
        AB = AB @ tf.stack([zeros, tf.eye(n)], axis=0)
        Qs = tf.matmul(AB[:, :n, :], Fs, transpose_b=True)
        return LGSSM(sde.P0, Fs, Qs, sde.H, R)