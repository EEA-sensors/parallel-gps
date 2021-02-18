# Regression experiments on sinusoidal signals.
# Corresponds to the *** of paper.
import os

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from gpflow.models import GPModel
from tensorflow_probability.python.distributions import Normal

from pssgp.experiments.common import ModelEnum, get_model, \
    run_one_mcmc, MCMC
from pssgp.experiments.sunspot.common import get_data, FLAGS
from pssgp.kernels import Matern32

flags.DEFINE_integer('np_seed', 666, "data model seed")
flags.DEFINE_integer('tf_seed', 31415, "mcmc model seed")
flags.DEFINE_integer('n_runs', 10, "size of the logspace for n training samples")
flags.DEFINE_string('mcmc', MCMC.HMC.value, "MCMC method enum")
flags.DEFINE_integer('n_samples', 1000, "Number of samples required")
flags.DEFINE_integer('n_burnin', 100, "Number of burnin samples")
flags.DEFINE_float('step_size', 0.1, "Step size for the gradient based chain")
flags.DEFINE_float('n_leapfrogs', 10, "Num leapfrogs for HMC")

flags.DEFINE_boolean('plot', True, "Plot the result")
flags.DEFINE_boolean('run', True, "Run the result or load the data")


def set_gp_priors(gp_model: GPModel):
    gp_dtype = gpf.config.default_float()
    variance_prior = Normal(gp_dtype(FLAGS.noise_variance), gp_dtype(FLAGS.noise_variance))
    if FLAGS.model == ModelEnum.GP.value:
        gp_model.likelihood.variance.prior = variance_prior
    else:
        gp_model.noise_variance.prior = variance_prior


def get_covariance_function():
    gp_dtype = gpf.config.default_float()

    matern_variance = 5500.
    matern_lengthscales = 5.

    m32_cov = Matern32(variance=matern_variance, lengthscales=matern_lengthscales)

    m32_cov.variance.prior = Normal(gp_dtype(matern_variance), gp_dtype(matern_variance))
    m32_cov.lengthscales.prior = Normal(gp_dtype(matern_lengthscales), gp_dtype(matern_lengthscales))

    return m32_cov


def run():
    gpf.config.set_default_float(getattr(np, FLAGS.dtype))

    tf.random.set_seed(FLAGS.tf_seed)
    f_times = os.path.join("results", f"mcmc-times-{FLAGS.model}-{FLAGS.mcmc}")
    # TODO: we need a flag for this directory really.
    f_posterior = os.path.join("results", f"mcmc-posterior-{FLAGS.model}-{FLAGS.mcmc}")

    n_training_logspace = [3000, 300, 50]
    cov_fun = get_covariance_function()
    if FLAGS.run:
        times = np.empty(len(n_training_logspace), dtype=float)
        for i, n_training in tqdm.tqdm(enumerate(n_training_logspace), total=FLAGS.n_runs):
            t, y = get_data(n_training)
            gp_model = get_model(ModelEnum(FLAGS.model), (t, y), FLAGS.noise_variance, cov_fun,
                                 t.shape[0])
            set_gp_priors(gp_model)

            run_time, params_res = run_one_mcmc(n_training, gp_model)
            times[i] = run_time
            np.savez(f_posterior + f"-{n_training}", **params_res)
        np.save(f_times, np.stack([n_training_logspace, times], axis=1))
    if FLAGS.plot:
        T = 2000
        for n_training in n_training_logspace:
            result = np.load(f_posterior + f"-{n_training}.npz")
            t, y = get_data(n_training)
            rng = np.random.RandomState(FLAGS.np_seed)
            interpolation_times = np.sort(rng.uniform(t[0], t[-1], 2000))[:, None]
            gp_model = get_model(ModelEnum(FLAGS.model), (t, y), FLAGS.noise_variance, cov_fun,
                                 T + n_training)
            ax = plt.subplot()
            ax.scatter(t[:, ], y[:, 0], s=1, marker="x", color="k")
            for param_name, sample in result.items():
                param = eval(f"gp_model{param_name}")
                print(param, np.mean(sample))

            for i in rng.choice(FLAGS.n_samples, 10, replace=True):
                for param, sample in zip(gp_model.trainable_variables, result.values()):
                    # param = eval(f"gp_model{param_name}")
                    param.assign(sample[i])
                interpolated_points, interpolation_cov = gp_model.predict_f(interpolation_times)
                ax.plot(interpolation_times[:, 0], interpolated_points[:, 0], alpha=0.25, color="blue")

            plt.show()


def main(_):
    device = tf.device(FLAGS.device)
    with device:
        run()


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs('results')
    app.run(main)
