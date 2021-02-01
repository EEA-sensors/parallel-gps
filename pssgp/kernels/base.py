import abc
from collections import namedtuple

import tensorflow as tf
from gpflow import config

from pssgp.kalman.base import LGSSM

ContinuousDiscreteModel = namedtuple("ContinuousDiscreteModel", ["P0", "F", "L", "H", "Q"])


@tf.function
def _get_ssm(sde, ts, R, t0=0.):
    dtype = config.default_float()
    n = tf.shape(sde.F)[0]
    t0 = tf.reshape(tf.cast(t0, dtype), (1, 1))

    ts = tf.concat([t0, ts], axis=0)
    dts = tf.reshape(ts[1:] - ts[:-1], (-1, 1, 1))
    Fs = tf.linalg.expm(dts * tf.expand_dims(sde.F, 0))
    zeros = tf.zeros_like(sde.F)

    Phi = tf.concat(
        [tf.concat([sde.F, sde.L @ tf.matmul(sde.Q, sde.L, transpose_b=True)], axis=1),
         tf.concat([zeros, -tf.transpose(sde.F)], axis=1)],
        axis=0)

    AB = tf.linalg.expm(dts * tf.expand_dims(Phi, 0))
    AB = AB @ tf.concat([zeros, tf.eye(n, dtype=dtype)], axis=0)
    Qs = tf.matmul(AB[:, :n, :], Fs, transpose_b=True)
    return LGSSM(sde.P0, Fs, Qs, sde.H, R)


def solve_lyap_vec(F: tf.Tensor,
                   L: tf.Tensor,
                   q: tf.Tensor) -> tf.Tensor:
    """Vectorized Lyapunov equation solver

    F P + P F' + L q L' = 0

    Parameters
    ----------
    F : tf.Tensor
        ...
    L : tf.Tensor
        ...
    q : tf.Tensor
        ...

    Returns
    -------
    Pinf : tf.Tensor
        Steady state covariance

    """
    dtype = config.default_float()

    dim = tf.shape(F)[0]

    op1 = tf.linalg.LinearOperatorFullMatrix(F)
    op2 = tf.linalg.LinearOperatorIdentity(dim, dtype=dtype)

    F1 = tf.linalg.LinearOperatorKronecker([op2, op1]).to_dense()
    F2 = tf.linalg.LinearOperatorKronecker([op1, op2]).to_dense()

    F = F1 + F2

    Q = L @ tf.transpose(L) * q

    Pinf = tf.reshape(tf.linalg.solve(F, tf.reshape(Q, (-1, 1))), (dim, dim))
    Pinf = -0.5 * (Pinf + tf.transpose(Pinf))
    return Pinf


class SDEKernelMixin(metaclass=abc.ABCMeta):
    def __init__(self, t0: float = 0.):
        """

        Parameters:
        -----------
        t0: float, optional
        rbf_order : int, default=6
            The order of Taylor expansion for RBF covariance function in state-space
        """
        self.t0 = t0

    @abc.abstractmethod
    def get_sde(self) -> ContinuousDiscreteModel:
        """
        Creates the linear time invariant continuous discrete system associated to the stationary kernel at hand

        Returns
        -------
        sde: ContinuousDiscreteModel
            The associated LTI model
        """

    def get_ssm(self, ts, R, t0=0.):
        """
        Creates the linear Gaussian state space model associated to the stationary kernel at hand

        Parameters
        ----------
        ts: tf.Tensor
            The times at which we have observations
        R: tf.Tensor
            The observation covariance
        t0: float
            Starting point of the model

        Returns
        -------
        lgssm: ContinuousDiscreteModel
            The associated state space model
        """
        ssm = _get_ssm(self.get_sde(), ts, R, t0)
        return ssm
