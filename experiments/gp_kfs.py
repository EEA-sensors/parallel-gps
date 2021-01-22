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
y = obs_noise(ft, 0.01 * np.eye(t.shape[0]))

# Init cov function
cov_func = SSMatern32(variance=1,
                      lengthscales=0.5)

# Init regression model
# m = gpf.models.GPR(data=(np.reshape(t, (500, 1)), np.reshape(y, (500, 1))),
#                    kernel=cov_func,
#                    mean_function=None)
m = StateSpaceGP(data=(np.reshape(t, (500, 1)), np.reshape(y, (500, 1))),
                 kernel=cov_func,
                 noise_variance=0.01)

# Hyperparas
# m.likelihood.variance.assign(0.01)
# m.kernel.lengthscales.assign(0.5)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
# print(m.trainable_variables)
# with tf.GradientTape() as tape:
#     tape.watch(m.trainable_variables)
#     loss = m._training_loss()
# print(tape.gradient(loss, m.trainable_variables))
opt_logs = opt.minimize(m._training_loss, m.trainable_variables, options=dict(maxiter=100))

# Prediction
query = np.sort(np.random.rand(800)).reshape(800, 1)

mean, var = m.predict_f(query)

print(mean)

plt.plot(t, ft, c='k')
plt.scatter(t, y, c='r')
plt.plot(query, mean[:, 0], c='g')

plt.show()
