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

T = 1000
K = 800
t = np.sort(np.random.rand(T))
ft = sinu(t)
y = obs_noise(ft, 0.01 * np.eye(1))
# Init cov function
cov_func = SSMatern32(variance=1,
                      lengthscales=1)

# Init regression model
# m = gpf.models.GPR(data=(np.reshape(t, (500, 1)), np.reshape(y, (500, 1))),
#                    kernel=cov_func,
#                    mean_function=None)
m = StateSpaceGP(data=(np.reshape(t, (T, 1)), np.reshape(y, (T, 1))),
                 kernel=cov_func,
                 noise_variance=0.01)

# Hyperparas
# m.likelihood.variance.assign(0.01)
m.kernel.lengthscales.assign(0.5)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
# print(m.trainable_variables)
# with tf.GradientTape() as tape:
#     tape.watch(m.trainable_variables)
#     loss = m._training_loss()
# print(tape.gradient(loss, m.trainable_variables))
opt_logs = opt.minimize(m._training_loss, m.trainable_variables, options=dict(maxiter=100))

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
