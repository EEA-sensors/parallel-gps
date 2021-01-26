import numpy.testing as npt

import gpflow as gpf
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.kernels import RBF, Matern52
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise

# Generate data
tf.random.set_seed(666)
np.random.seed(666)

T = 100
K = 200
t = np.linspace(0, 10, T)
ft = sinu(t)
y = obs_noise(ft, 0.1 * np.eye(1))

cov_func = Matern52(variance=1.,
               lengthscales=0.1)

# Init regression model
m = StateSpaceGP(data=(np.reshape(t, (T, 1)), np.reshape(y, (T, 1))),
                 kernel=cov_func,
                 noise_variance=0.1)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
opt_logs = opt.minimize(m._training_loss, m.trainable_variables, options=dict(maxiter=100))

# Prediction
query = np.linspace(0, 10, K).reshape(K, 1)

mean, var = m.predict_f(query)

plt.plot(t, ft, c='k')
# plt.scatter(t, y, c='r')
plt.plot(query.squeeze(), mean[:, 0], c='g')

plt.fill_between(
    query[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0, 0]),
    color="C0",
    alpha=0.2,
)
plt.suptitle("SSGP")

plt.show()