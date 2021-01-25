"""
Numerically test if GP reg has the same results with KFS
"""

import time
import unittest
import numpy.testing as npt

import gpflow as gpf
import numpy as np
import tensorflow as tf

from src.kernels.matern import Matern12, Matern32, Matern52
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise


class GPEquivalenceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.T = 2000
        self.K = 800
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.1 * np.eye(1))

        self.covs = (Matern12(variance=1, lengthscales=0.5),
                     Matern32(variance=1, lengthscales=0.5),
                     Matern52(variance=1, lengthscales=0.5))

    def test_loglikelihood(self):

        for cov in self.covs:

            gp_model = gpf.models.GPR(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                      kernel=cov,
                                      noise_variance=0.1,
                                      mean_function=None)

            ss_model = StateSpaceGP(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                    kernel=cov,
                                    noise_variance=0.1)

            npt.assert_almost_equal(gp_model.log_marginal_likelihood(),
                                    ss_model.maximum_log_likelihood_objective(),
                                    decimal=6)

    def test_posterior(self):

        for cov in self.covs:

            gp_model = gpf.models.GPR(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                      kernel=cov,
                                      noise_variance=0.1,
                                      mean_function=None)

            ss_model = StateSpaceGP(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                    kernel=cov,
                                    noise_variance=0.1)

            query = np.sort(np.random.rand(self.K)).reshape(self.K, 1)

            mean_gp, var_gp = gp_model.predict_f(query)

            mean_ss, var_ss = ss_model.predict_f(query)

            npt.assert_array_almost_equal(mean_gp[:, 0], mean_ss[:, 0], decimal=8)

if __name__ == '__main__':
    unittest.main()
