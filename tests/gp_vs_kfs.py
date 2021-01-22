"""
Numerically test if GP reg has the same results with KFS
"""

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.kernels.matern import Matern32 as SSMatern32
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise

# Generate data
tf.random.set_seed(666)
np.random.seed(666)

t = np.sort(np.random.rand(500))
ft = sinu(t)
y = obs_noise(ft, 0.1 * np.eye(t.shape[0]))

# Init cov functions
ss_cov = SSMatern32(variance=1,
                    lengthscales=0.5)

gp_cov = gpf.kernels.Matern32(variance=1,
                              lengthscales=0.5)

gp_model = gpf.models.GPR(data=(np.reshape(t, (500, 1)), np.reshape(y, (500, 1))),
                          kernel=gp_cov,
                          noise_variance=0.1,
                          mean_function=None)
ss_model = StateSpaceGP(data=(np.reshape(t, (500, 1)), np.reshape(y, (500, 1))),
                        kernel=ss_cov,
                        noise_variance=0.1)

# Prediction
query = np.sort(np.random.rand(800)).reshape(800, 1)

# Precition from GP
mean_gp, var_gp = gp_model.predict_f(query)

# Precition from KFS
mean_ss, var_ss = ss_model.predict_f(query)

plt.plot(query, mean_gp, c='k', label='GP')
plt.plot(query, mean_ss[:, 0], c='r', label='KFS')
plt.legend()
plt.show()