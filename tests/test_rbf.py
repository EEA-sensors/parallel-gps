import unittest
import numpy.testing as npt

import gpflow as gpf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.kernels import RBF
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise


class RBFTest(unittest.TestCase):

    def setUp(self):
        self.T = 2000
        self.K = 800
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.01)

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

    def test_loglikelihood(self):
        rbf_order = 12
        cov = RBF(variance=1., lengthscales=0.1, order=rbf_order)

        gp_model = gpf.models.GPR(data=self.data,
                                  kernel=cov,
                                  noise_variance=0.1,
                                  mean_function=None)
        gp_model_ll = gp_model.log_marginal_likelihood()
        for parallel in [False, True]:
            ss_model = StateSpaceGP(data=self.data,
                                    kernel=cov,
                                    noise_variance=0.1,
                                    parallel=parallel)
            ss_model_ll = ss_model.maximum_log_likelihood_objective()
            npt.assert_allclose(gp_model_ll,
                                ss_model_ll,
                                atol=1e-3,
                                rtol=1e-3)

    def test_posterior(self):
        rbf_order = 12
        cov = RBF(variance=1., lengthscales=0.1, order=rbf_order)

        query = np.sort(np.random.rand(self.K)).reshape(self.K, 1)

        gp_model = gpf.models.GPR(data=self.data,
                                  kernel=cov,
                                  noise_variance=0.1,
                                  mean_function=None)
        mean_gp, var_gp = gp_model.predict_f(query)

        for parallel in [False, True]:
            ss_model = StateSpaceGP(data=self.data,
                                    kernel=cov,
                                    noise_variance=0.1,
                                    parallel=parallel)
            mean_ss, var_ss = ss_model.predict_f(query)

            npt.assert_allclose(mean_gp[:, 0], mean_ss[:, 0], atol=1e-3, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
