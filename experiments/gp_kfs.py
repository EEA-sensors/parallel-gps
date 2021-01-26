import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.kernels import Matern32, Matern52, Matern12
from src.model import StateSpaceGP
from src.toymodels import sinu, obs_noise

# Generate data
tf.random.set_seed(666)
np.random.seed(666)

T = 1000
K = 800
t = np.sort(np.random.rand(T))
ft = sinu(t)
y = obs_noise(ft, 0.01)
# Init cov function
cov_func = Matern32(variance=1.,
                    lengthscales=0.1)


# Init regression model
data = (tf.constant(t[:, None]), tf.constant(y[:, None]))
m = StateSpaceGP(data=data,
                 kernel=cov_func,
                 noise_variance=0.01)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
loss = lambda : -m.log_posterior_density()
opt_logs = opt.minimize(loss, m.trainable_variables, options=dict(maxiter=100))

# Prediction
query = np.sort(np.random.rand(K)).reshape(K, 1)

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
