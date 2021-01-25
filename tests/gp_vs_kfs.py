"""
Numerically test if GP reg has the same results with KFS
"""

import unittest

import gpflow as gpf
import numpy as np
import numpy.testing as npt

from src.kernels.matern import Matern12, Matern32, Matern52
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise


# def tf_function(func=None,
#                  input_signature=None,
#                  autograph=True,
#                  experimental_implements=None,
#                  experimental_autograph_options=None,
#                  experimental_relax_shapes=False,
#                  experimental_compile=None,
#                  experimental_follow_type_hints=None):
#     if func is not None:
#         return func
#     else:
#         return lambda x: x
#
# tf.function = tf_function


class GPEquivalenceTest(unittest.TestCase):

    def setUp(self):
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
            gp_model_ll = gp_model.log_marginal_likelihood()
            for parallel in [False, True]:
                ss_model = StateSpaceGP(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                        kernel=cov,
                                        noise_variance=0.1,
                                        parallel=parallel)
                ss_model_ll = ss_model.maximum_log_likelihood_objective()
                npt.assert_almost_equal(gp_model_ll,
                                        ss_model_ll,
                                        decimal=6)

    def test_posterior(self):
        query = np.sort(np.random.rand(self.K)).reshape(self.K, 1)
        for cov in self.covs:
            gp_model = gpf.models.GPR(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                      kernel=cov,
                                      noise_variance=0.1,
                                      mean_function=None)
            mean_gp, var_gp = gp_model.predict_f(query)
            for parallel in [False, True]:
                ss_model = StateSpaceGP(data=(np.reshape(self.t, (self.T, 1)), np.reshape(self.y, (self.T, 1))),
                                        kernel=cov,
                                        noise_variance=0.1,
                                        parallel=parallel)
                mean_ss, var_ss = ss_model.predict_f(query)
                npt.assert_array_almost_equal(mean_gp[:, 0], mean_ss[:, 0], decimal=8)


if __name__ == '__main__':
    unittest.main()
