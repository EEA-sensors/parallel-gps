"""
Numerically test if GP reg has the same results with KFS
"""

import time
import unittest

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.kernels.matern import Matern52
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise


class GPEquivalenceTest(unittest.TestCase):
    def __init__(self):
        pass

    def test_loglikelihood(self):
        pass

    def test_posterior(self):
        pass


# Generate data
tf.random.set_seed(666)
np.random.seed(666)

T = 2000
K = 800
t = np.sort(np.random.rand(T))
ft = sinu(t)
y = obs_noise(ft, 0.1 * np.eye(1))
# Init cov functions
cov = Matern52(variance=1,
               lengthscales=0.5)

gp_model = gpf.models.GPR(data=(np.reshape(t, (T, 1)), np.reshape(y, (T, 1))),
                          kernel=cov,
                          noise_variance=0.1,
                          mean_function=None)

ss_model = StateSpaceGP(data=(np.reshape(t, (T, 1)), np.reshape(y, (T, 1))),
                        kernel=cov,
                        noise_variance=0.1)

# Prediction
query = np.sort(np.random.rand(K)).reshape(K, 1)

# Precition from GP
mean_gp, var_gp = gp_model.predict_f(query)
tic = time.time()
mean_gp, var_gp = gp_model.predict_f(query)
print(f"Conventional GP: {time.time() - tic}")
# Precition from KFS
mean_ss, var_ss = ss_model.predict_f(query)

tic = time.time()
mean_ss, var_ss = ss_model.predict_f(query)
print(f"State space GP: {time.time() - tic}")

plt.plot(query, mean_ss[:, 0] - mean_gp[:, 0], c='k', label='GP')
# plt.plot(query, mean_gp[:, 0], c='k', label='GP')
# plt.plot(query, mean_ss[:, 0], c='r', label='KFS')
plt.legend()
plt.show()
