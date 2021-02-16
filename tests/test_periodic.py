import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from gpflow.kernels import SquaredExponential
from pssgp.kernels import Periodic
from pssgp.toymodels import sinu, obs_noise
from pssgp.kernels.periodic import _get_offline_coeffs


class PeriodicTest(unittest.TestCase):

    def setUp(self):
        seed = 31415926
        self.T = 2000
        self.K = 800
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.01, seed)

        self.data = (tf.constant(self.t[:, None]), tf.constant(self.y[:, None]))

        periodic_order = 2
        periodic_base_kernel = SquaredExponential(variance=1., lengthscales=0.1)
        self.cov = Periodic(periodic_base_kernel, period=1., order=periodic_order)

    def test_offline_coeffs(self):
        b, K, div_facto_K = _get_offline_coeffs(2)

        npt.assert_almost_equal(b, np.array([[1, 0, 0],
                                             [0, 2, 0],
                                             [2, 0, 2]]), decimal=8)
        npt.assert_almost_equal(K, np.array([[0, 0, 0],
                                             [1, 1, 1],
                                             [2, 2, 2]]), decimal=8)
        npt.assert_almost_equal(div_facto_K, np.array([[1, 1, 1],
                                                       [1, 1, 1],
                                                       [0.5, 0.5, 0.5]]), decimal=8)

    def test_sde_coeff(self):
        F_expected = np.zeros((6, 6))
        F_expected[2, 3] = -6.283185307179586
        F_expected[4, 5] = -12.5663706143592
        F_expected = F_expected - F_expected.T

        H_expected = np.array([[1, 0, 1, 0, 1, 0]])
        L_expected = np.eye(6)
        Q_expected = np.zeros((6, 6))

        Pinf_expected = np.diag([1.20739740482544e-19, 1.20739740482544e-19, 9.64374923981979e-21,
                                 9.64374923981979e-21, 1.20546865497747e-19, 1.20546865497747e-19])

        Pinf, F, L, H, Q = self.cov.get_sde()

        npt.assert_almost_equal(F, F_expected)
        npt.assert_almost_equal(L, L_expected)
        npt.assert_almost_equal(H, H_expected)
        npt.assert_almost_equal(Q, Q_expected)
        npt.assert_almost_equal(Pinf, Pinf_expected)


if __name__ == '__main__':
    unittest.main()
