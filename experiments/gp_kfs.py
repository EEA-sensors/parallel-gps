import warnings

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gpflow.models import GPR

from pssgp.kernels import RBF
from pssgp.model import StateSpaceGP
from pssgp.toymodels import sinu, obs_noise

# Generate data
tf.random.set_seed(666)
np.random.seed(666)

use_GPR = False

T = 1000
K = 800
t = np.sort(np.random.rand(T))
ft = sinu(t)
y = obs_noise(ft, 0.01)
# Init cov function
cov_func = RBF(variance=1.,
               lengthscales=0.1,
               order=12)

# Init regression model
data = (tf.constant(t[:, None]), tf.constant(y[:, None]))
if use_GPR:
    m = GPR(data=data,
            kernel=cov_func,
            noise_variance=0.01,
            mean_function=None)
else:
    m = StateSpaceGP(data=data,
                     kernel=cov_func,
                     noise_variance=0.01,
                     parallel=True)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
loss = lambda: -m.log_posterior_density()

opt_logs = opt.minimize(loss, m.trainable_variables, options=dict(maxiter=100))

if use_GPR:
    print("obs noise: ", tf.nn.softplus(m.likelihood.variance))
else:
    print("obs noise: ", tf.nn.softplus(m._noise_variance))
print("ell: ", tf.nn.softplus(cov_func.lengthscales))
print("var: ", tf.nn.softplus(cov_func.variance))
# Prediction
query = np.sort(np.random.rand(K)).reshape(K, 1)

mean, var = m.predict_f(query)

plt.plot(t, ft, c='k')
# plt.scatter(t, y, c='r')
plt.plot(query.squeeze(), mean[:, 0], c='g')

# plt.fill_between(
#     query[:, 0],
#     mean[:, 0] - 1.96 * np.sqrt(np.squeeze(var[:, 0])),
#     mean[:, 0] + 1.96 * np.sqrt(np.squeeze(var[:, 0])),
#     color="C0",
#     alpha=0.2,
# )
if use_GPR:
    plt.suptitle("GPR")
else:
    plt.suptitle("SSGP")

plt.show()
