import unittest

import numpy as np
import numpy.testing as npt
import tensorflow as tf

from pssgp.kernels import RBF
from pssgp.toymodels import sinu, obs_noise


class RBFTest(unittest.TestCase):

    def setUp(self):
        seed = 31415926
        self.T = 2000
        self.K = 800
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.01, seed)

        self.data = (tf.constant(self.t[:, None]), tf.constant(self.y[:, None]))

        rbf_order = 3
        self.cov = RBF(variance=1., lengthscales=0.1, order=rbf_order)

    def test_sde_coefficients(self):
        F_expected = np.array([[0, 14.520676967550859, 0],
                               [0, 0, 32.857489440296360],
                               [-14.5210953665873, -29.4746060478111, -50.3678777987092]])

        L_expected = np.array([0., 0., 1.]).reshape(3, 1)

        H_expected = np.array([1., 0., 0.]).reshape(1, 3)

        Q_expected = 52.8553179255264

        Pinf_expected = np.array([[1.04502531824891, -1.41636387123970e-17, -0.301281550265743],
                                  [-1.41636387123970e-17, 0.681741999944955, -1.70331397804495e-17],
                                  [-0.301281550265743, -1.70331397804495e-17, 0.611552410634913]])

        Pinf, F, L, H, Q = self.cov.get_sde()

        npt.assert_almost_equal(F, F_expected, decimal=8)
        npt.assert_almost_equal(L, L_expected, decimal=8)
        npt.assert_almost_equal(H, H_expected, decimal=8)
        npt.assert_almost_equal(Q, Q_expected, decimal=8)
        npt.assert_almost_equal(Pinf, Pinf_expected, decimal=8)


if __name__ == '__main__':
    unittest.main()
