"""
Numerically test if GP reg has the same results with KFS
"""

import unittest

import gpflow as gpf
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from pssgp.kernels import RBF
from pssgp.kernels.base import SDESum
from pssgp.kernels.matern import Matern12, Matern32, Matern52
from pssgp.model import StateSpaceGP
from pssgp.toymodels import sinu, obs_noise


def setUpModule():  # noqa: unittest syntax.
    # goal is to test the logic, not the runtime.
    np.random.seed(666)
    tf.config.set_visible_devices([], 'GPU')


class GPEquivalenceTest(unittest.TestCase):

    def setUp(self):
        self.T = 100
        self.K = 50
        self.t = np.sort(np.random.rand(self.T))
        self.ft = sinu(self.t)
        self.y = obs_noise(self.ft, 0.1)
        self.covs = (
            (Matern12(variance=1., lengthscales=0.5), 1e-6, 1e-2),
            (Matern32(variance=1., lengthscales=0.5), 1e-6, 1e-2),
            (Matern52(variance=1., lengthscales=0.5), 1e-6, 1e-2),
            (RBF(variance=1., lengthscales=0.5, order=15, balancing_iter=10), 1e-2, 1e-2),
        )
        self.covs += ((self.covs[0][0] + self.covs[1][0], 1e-6, 1e-2),)  # whatever that means, just testing the sum
        self.covs += ((self.covs[0][0] * self.covs[1][0], 1e-6, 1e-2),)  # whatever that means, just testing the prod

        self.data = (tf.constant(self.t[:, None]), tf.constant(self.y[:, None]))

    def test_loglikelihood(self):
        for cov, val_tol, grad_tol in self.covs:
            if not isinstance(cov, SDESum):
                continue
            check_grad_vars = cov.trainable_variables

            gp_model = gpf.models.GPR(data=self.data,
                                      kernel=cov,
                                      noise_variance=0.1,
                                      mean_function=None)
            with tf.GradientTape() as tape:
                tape.watch(check_grad_vars)
                gp_model_ll = gp_model.maximum_log_likelihood_objective()
            gp_model_grad = tape.gradient(gp_model_ll, check_grad_vars)

            for parallel in [True]:
                ss_model = StateSpaceGP(data=self.data,
                                        kernel=cov,
                                        noise_variance=0.1,
                                        parallel=parallel)
                with tf.GradientTape() as tape:
                    tape.watch(check_grad_vars)
                    ss_model_ll = ss_model.maximum_log_likelihood_objective()
                ss_model_grad = tape.gradient(ss_model_ll, check_grad_vars)
                npt.assert_allclose(gp_model_ll,
                                    ss_model_ll,
                                    atol=val_tol,
                                    rtol=val_tol)
                for gp_grad, ss_grad in zip(gp_model_grad, ss_model_grad):
                    npt.assert_allclose(gp_grad,
                                        ss_grad,
                                        atol=grad_tol,
                                        rtol=grad_tol)

    @unittest.skip
    def test_posterior(self):
        query = tf.constant(np.sort(np.random.rand(self.K, 1), 0))
        for cov, val_tol, _ in self.covs:
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
                npt.assert_allclose(mean_gp[:, 0], mean_ss[:, 0], atol=val_tol, rtol=val_tol)


if __name__ == '__main__':
    unittest.main()
