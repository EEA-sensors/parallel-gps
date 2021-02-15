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


def get_lssm_spec(dim, T):
    dtype = config.default_float()
    P0_spec = tf.TensorSpec((dim, dim), dtype=dtype)
    Fs_spec = tf.TensorSpec((T, dim, dim), dtype=dtype)
    Qs_spec = tf.TensorSpec((T, dim, dim), dtype=dtype)
    H_spec = tf.TensorSpec((1, dim), dtype=dtype)
    R_spec = tf.TensorSpec((1, 1), dtype=dtype)

    return LGSSM(P0_spec, Fs_spec, Qs_spec, H_spec, R_spec)


def _get_ssm(sde, ts, R, t0=0.):
    dtype = config.default_float()
    n = tf.shape(sde.F)[0]
    t0 = tf.reshape(tf.convert_to_tensor(t0, dtype=dtype), (1, 1))

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
    def __init__(self, t0: float = 0., **_kwargs):
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
        return SDESum([self, other])  # noqa: don't complain Pycharm, I know what's good for you.

    def __mul__(self, other):
        return SDEProduct([self, other])  # noqa: don't complain Pycharm, I know what's good for you.

    @abc.abstractmethod
    def get_spec(self, T):
        return None


def _sde_combination_init(self, kernels: List[Kernel], name: Optional[str] = None, **kargs):
    if not all(isinstance(k, SDEKernelMixin) for k in kernels):
        raise TypeError("can only combine SDE Kernel instances")  # pragma: no cover
    gpflow.kernels.Sum.__init__(self, kernels, name)
    SDEKernelMixin.__init__(self, **kargs)


def block_diag(arrs):
    xdims = [tf.shape(a)[0] for a in arrs]
    ydims = [tf.shape(a)[1] for a in arrs]
    out_dtype = arrs[0].dtype
    out = tf.zeros((0, sum(ydims)), dtype=out_dtype)
    ydim = sum(ydims)
    r, c = 0, 0
    for i, (rr, cc) in enumerate(zip(xdims, ydims)):
        paddings = [[0, 0],
                    [c, ydim - c - cc]]

        out = tf.concat([out, tf.pad(arrs[i], paddings)], 0)
        r = r + rr
        c = c + cc
    return out


class SDESum(SDEKernelMixin, gpflow.kernels.Sum):
    __init__ = _sde_combination_init

    def get_spec(self, T):
        dim = 0
        for kernel in self.kernels:
            ker_spec = kernel.get_spec(T)
            if ker_spec is None:
                return None
            ker_P0_spec = ker_spec.P0
            dim += ker_P0_spec.shape[-1]
        return get_lssm_spec(dim, T)

    @staticmethod
    def _block_diagonal(matrices, is_positive_definite=False, square=True):
        if square:
            operators = [tf.linalg.LinearOperatorFullMatrix(matrix, is_positive_definite) for matrix in matrices]
            block_op = tf.linalg.LinearOperatorBlockDiag(operators)
            return block_op.to_dense()
        return block_diag(matrices)

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
        return ContinuousDiscreteModel(self._block_diagonal(P0s, is_positive_definite=True),
                                       self._block_diagonal(Fs),
                                       self._block_diagonal(Ls, square=False),
                                       tf.concat(Hs, axis=1),
                                       self._block_diagonal(Qs, is_positive_definite=True))


class SDEProduct(SDEKernelMixin, gpflow.kernels.Product):
    __init__ = _sde_combination_init

    def get_spec(self, T):
        dim = 1
        for kernel in self.kernels:
            ker_spec = kernel.get_spec(T)
            if ker_spec is None:
                return None
            ker_P0_spec = ker_spec.P0
            dim *= ker_P0_spec.shape[-1]
        return get_lssm_spec(dim, T)

    @staticmethod
    def _combine_F(op1, op2):
        I1 = tf.linalg.LinearOperatorIdentity(tf.shape(op1)[0], dtype=op1.dtype)
        I2 = tf.linalg.LinearOperatorIdentity(tf.shape(op2)[0], dtype=op2.dtype)
        op1 = tf.linalg.LinearOperatorFullMatrix(op1)
        op2 = tf.linalg.LinearOperatorFullMatrix(op2)
        kron_1 = tf.linalg.LinearOperatorKronecker([op1, I2])
        kron_2 = tf.linalg.LinearOperatorKronecker([I1, op2])
        return kron_1.to_dense() + kron_2.to_dense()

    @staticmethod
    def _combine_Q(sde1, sde2):
        gamma1 = tf.linalg.LinearOperatorFullMatrix(sde1.L @ sde1.Q @ tf.transpose(sde1.L), is_positive_definite=True,
                                                    is_self_adjoint=True)
        gamma2 = tf.linalg.LinearOperatorFullMatrix(sde2.L @ sde2.Q @ tf.transpose(sde2.L), is_positive_definite=True,
                                                    is_self_adjoint=True)
        Pinf1 = tf.linalg.LinearOperatorFullMatrix(sde1.P0, is_positive_definite=True, is_self_adjoint=True)
        Pinf2 = tf.linalg.LinearOperatorFullMatrix(sde2.P0, is_positive_definite=True, is_self_adjoint=True)

        kron_1 = tf.linalg.LinearOperatorKronecker([gamma1, Pinf2])
        kron_2 = tf.linalg.LinearOperatorKronecker([Pinf1, gamma2])
        return kron_1.to_dense() + kron_2.to_dense()

    def get_sde(self) -> ContinuousDiscreteModel:
        """
        Creates the linear time invariant continuous discrete system associated to the stationary kernel at hand

        Returns
        -------
        sde: ContinuousDiscreteModel
            The associated LTI model
        """
        dtype = gpflow.config.default_float()
        kernels = self.kernels  # type: List[SDEKernelMixin]

        sdes = [kernel.get_sde() for kernel in kernels]

        F = reduce(self._combine_F, [sde.F for sde in sdes])
        Q = reduce(self._combine_Q, [sde for sde in sdes])
        P0 = tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(sde.P0, is_positive_definite=True,
                                                                                   is_self_adjoint=True)
                                                for sde in sdes]).to_dense()
        H = tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(sde.H)
                                               for sde in sdes]).to_dense()
        L = tf.eye(tf.shape(Q)[0], dtype=dtype)

        return ContinuousDiscreteModel(P0, F, L, H, Q)
