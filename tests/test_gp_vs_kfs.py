"""
Numerically test if GP reg has the same results with KFS
"""

import unittest

import gpflow as gpf
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from src.kernels.matern import Matern12, Matern32, Matern52
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise


class GPEquivalenceTest(unittest.TestCase):

    def setUp(self):
        self.T = 100
        self.K = 50
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.1)
        self.covs = (Matern12(variance=1., lengthscales=0.5),
                     Matern32(variance=1., lengthscales=0.5),
                     Matern52(variance=1., lengthscales=0.5))

        self.data = (tf.constant(self.t[:, None]), tf.constant(self.y[:, None]))

    @unittest.skip
    def test_loglikelihood(self):
        for cov in self.covs:
            gp_model = gpf.models.GPR(data=self.data,
                                      kernel=cov,
                                      noise_variance=0.1,
                                      mean_function=None)
            gp_model_ll = gp_model.log_marginal_likelihood()
            for parallel in [False, True]:
                ss_model = StateSpaceGP(data=self.data,
                                        kernel=cov,
                                        noise_variance=0.1,
                                        parallel=parallel, max_parallel=2 * (self.T + self.K))
                ss_model_ll = ss_model.maximum_log_likelihood_objective()
                npt.assert_almost_equal(gp_model_ll,
                                        ss_model_ll,
                                        decimal=6)

    def test_posterior(self):
        query = tf.constant(np.sort(np.random.rand(self.K, 1), 0))
        for cov in self.covs:
            gp_model = gpf.models.GPR(data=self.data,
                                      kernel=cov,
                                      noise_variance=0.1,
                                      mean_function=None)
            mean_gp, var_gp = gp_model.predict_f(query)
            for parallel in [False, True]:
                ss_model = StateSpaceGP(data=self.data,
                                        kernel=cov,
                                        noise_variance=0.1,
                                        parallel=parallel,
                                        max_parallel=2 * (self.T + self.K))
                mean_ss, var_ss = ss_model.predict_f(query)
                npt.assert_array_almost_equal(mean_gp[:, 0], mean_ss[:, 0], decimal=8)


if __name__ == '__main__':
    unittest.main()
