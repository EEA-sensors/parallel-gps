import numpy as np
import gpflow as gpf
import tensorflow as tf

from ..src.toymodels import sinu, obs_noise

import numpy as np
import matplotlib.pyplot as plt

# Generate data
tf.random.set_seed(666)
np.random.seed(666)

t = np.linspace(0, 5, 500)
ft = sinu(t)
y = obs_noise(ft, 0.01*np.eye(t.shape[0]))

# Init cov function
cov_func = gpf.kernels.Matern32()

# Init regression model
m = gpf.models.GPR(data=(t, y), 
                   kernel=cov_func, 
                   mean_function=None)

# Hyperparas
m.likelihood.variance.assign(0.01)
m.kernel.lengthscales.assign(0.5)

# Hyperpara opt
opt = gpf.optimizers.Scipy()
opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=100))

# Prediction
query = np.linspace(0, 5, 800)

mean, var = m.predict_f(query)