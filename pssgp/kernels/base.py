import abc
from collections import namedtuple
from functools import reduce
from typing import List, Optional

import gpflow
import tensorflow as tf
from gpflow import config
from gpflow.kernels import Kernel

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

    def __add__(self, other):
        return SDESum([self, other])

    def __mul(self, other):
        return SDEProduct([self, other])


def _sde_combination_init(self, kernels: List[Kernel], name: Optional[str] = None):
    if not all(isinstance(k, SDEKernelMixin) for k in kernels):
        raise TypeError("can only combine SDE Kernel instances")  # pragma: no cover
    super().__init__(kernels, name)


class SDESum(gpflow.kernels.Sum, SDEKernelMixin):
    __init__ = _sde_combination_init

    @staticmethod
    def _block_diagonal(matrices):
        operators = [tf.linalg.LinearOperatorFullMatrix(matrix) for matrix in matrices]
        block_op = tf.linalg.LinearOperatorBlockDiag(operators)
        return block_op.to_dense()

    def get_sde(self) -> ContinuousDiscreteModel:
        """
        Creates the linear time invariant continuous discrete system associated to the stationary kernel at hand

        Returns
        -------
        sde: ContinuousDiscreteModel
            The associated LTI model
        """
        kernels = self.kernels  # type: List[SDEKernelMixin]
        P0s = []
        Fs = []
        Ls = []
        Hs = []
        Qs = []

        for kernel in kernels:
            P0, F, L, H, Q = kernel.get_sde()
            P0s.append(P0)
            Fs.append(F)
            Ls.append(L)
            Hs.append(H)
            Qs.append(Q)
        return ContinuousDiscreteModel(self._block_diagonal(P0s),
                                       self._block_diagonal(Fs),
                                       self._block_diagonal(Ls),
                                       tf.concat(Hs, axis=1),
                                       self._block_diagonal(Qs))


class SDEProduct(gpflow.kernels.Product, SDEKernelMixin):
    __init__ = _sde_combination_init
    _LOW_LIM = 1e-6

    @staticmethod
    def _combine_F(F1, F2):
        I1 = tf.eye(tf.shape(F1)[0], dtype=F1.dtype)
        I2 = tf.eye(tf.shape(F2)[0], dtype=F2.dtype)
        return kronecker(F1, I2) + kronecker(I1, F2)

    @classmethod
    def _combine_Q(cls, e1, e2):
        Q1, P01 = e1
        Q2, P02 = e2
        Q1_zero = tf.reduce_all(tf.abs(Q1) < cls._LOW_LIM)
        Q2_zero = tf.reduce_all(tf.abs(Q2) < cls._LOW_LIM)

        Q1 = tf.cond(Q1_zero, P01, Q1)
        Q2 = tf.cond(Q2_zero, P02, Q2)

        return kronecker(Q1, Q2)

    def get_sde(self) -> ContinuousDiscreteModel:
        """
        Creates the linear time invariant continuous discrete system associated to the stationary kernel at hand

        Returns
        -------
        sde: ContinuousDiscreteModel
            The associated LTI model
        """
        kernels = self.kernels  # type: List[SDEKernelMixin]

        sdes = [kernel.get_sde() for kernel in kernels]

        F = reduce(self._combine_F, [sde.F for sde in sdes])
        Q = reduce(self._combine_Q, [(sde.Q, sde.P0) for sde in sdes])
        P0 = reduce(kronecker, [sde.P0 for sde in sdes])
        H = reduce(kronecker, [sde.H for sde in sdes])
        L = reduce(kronecker, [sde.L for sde in sdes])

        return ContinuousDiscreteModel(P0, F, L, H, Q)
