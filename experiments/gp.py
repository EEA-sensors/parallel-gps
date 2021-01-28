import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pssgp.toymodels import sinu, obs_noise

# Generate data
tf.random.set_seed(666)
np.random.seed(666)

t = np.sort(np.random.rand(500))
ft = sinu(t)
y = obs_noise(ft, 0.01 * np.eye(t.shape[0]))

# Init cov function
cov_func = gpf.kernels.Matern32()

# Init regression model
m = gpf.models.GPR(data=(np.reshape(t, (500, 1)), np.reshape(y, (500, 1))),
                   kernel=cov_func,
                   mean_function=None)

# Hyperparas
m.likelihood.variance.assign(0.01)
m.kernel.lengthscales.assign(0.5)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

# Prediction
query = np.linspace(0, 1, 800).reshape(800, 1)

mean, var = m.predict_f(query)

plt.plot(t, ft, c="k")
# plt.scatter(t, y, c="r")
plt.plot(query, mean, c="g")

plt.fill_between(
    query[:, 0],
    mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
    mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
    color="C0",
    alpha=0.2,
)
plt.suptitle("GP")
plt.show()
