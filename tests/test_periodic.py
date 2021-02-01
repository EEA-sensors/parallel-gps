import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from gpflow.kernels import SquaredExponential
from pssgp.kernels import SDEPeriodic
from pssgp.toymodels import sinu, obs_noise
from pssgp.kernels.periodic import _get_offline_coeffs


class PeriodicTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(31415926)
        self.T = 2000
        self.K = 800
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.01)

        self.data = (tf.constant(self.t[:, None]), tf.constant(self.y[:, None]))

        periodic_order = 6
        periodic_base_kernel = SquaredExponential(variance=1., lengthscales=0.1)
        self.cov = SDEPeriodic(periodic_base_kernel, period=1., order=periodic_order)

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
        sdes = self.cov.get_sde()
        pass


if __name__ == '__main__':
    unittest.main()
