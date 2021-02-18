# Regression experiments on sinusoidal signals.
# Corresponds to the *** of paper.
import os

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from gpflow import set_trainable
from gpflow.kernels import SquaredExponential
from gpflow.models import GPModel
from tensorflow_probability.python.distributions import Normal

from pssgp.experiments.co2.common import get_data, FLAGS
from pssgp.experiments.common import ModelEnum, get_model, \
    run_one_mcmc, MCMC
from pssgp.kernels import Matern32, Periodic

flags.DEFINE_integer('np_seed', 42, "data model seed")
flags.DEFINE_integer('tf_seed', 31415, "mcmc model seed")
flags.DEFINE_integer('n_runs', 10, "size of the logspace for n training samples")
flags.DEFINE_string('mcmc', MCMC.HMC.value, "MCMC method enum")
flags.DEFINE_integer('n_samples', 1000, "Number of samples required")
flags.DEFINE_integer('n_burnin', 100, "Number of burnin samples")
flags.DEFINE_float('step_size', 0.01, "Step size for the gradient based chain")
flags.DEFINE_float('n_leapfrogs', 10, "Num leapfrogs for HMC")

flags.DEFINE_boolean('plot', False, "Plot the result")
flags.DEFINE_boolean('run', True, "Run the result or load the data")




def set_gp_priors(gp_model: GPModel):
    if FLAGS.model == ModelEnum.GP.value:
        set_trainable(gp_model.likelihood.variance, False)
    else:
        set_trainable(gp_model.noise_variance, False)


def get_covariance_function():
    gp_dtype = gpf.config.default_float()
    # Matern 32
    m32_cov = Matern32(variance=1, lengthscales=100.)
    m32_cov.variance.prior = Normal(gp_dtype(1.), gp_dtype(0.1))
    m32_cov.lengthscales.prior = Normal(gp_dtype(100.), gp_dtype(50.))

    # Periodic base kernel
    periodic_base_cov = SquaredExponential(variance=5., lengthscales=1.)
    set_trainable(periodic_base_cov.variance, False)
    periodic_base_cov.lengthscales.prior = Normal(gp_dtype(5.), gp_dtype(1.))

    # Periodic
    periodic_cov = Periodic(periodic_base_cov, period=1., order=FLAGS.qp_order)
    set_trainable(periodic_cov.period, False)

    # Periodic damping
    periodic_damping_cov = Matern32(variance=1e-1, lengthscales=50)
    periodic_damping_cov.variance.prior = Normal(gp_dtype(1e-1), gp_dtype(1e-3))
    periodic_damping_cov.lengthscales.prior = Normal(gp_dtype(50), gp_dtype(10.))

    # Final covariance
    co2_cov = periodic_cov * periodic_damping_cov + m32_cov
    return co2_cov


def run():
    gpf.config.set_default_float(getattr(np, FLAGS.dtype))

    tf.random.set_seed(FLAGS.tf_seed)
    f_times = os.path.join("results", f"mcmc-times-{FLAGS.model}-{FLAGS.mcmc}")
    # TODO: we need a flag for this directory really.
    f_posterior = os.path.join("results", f"mcmc-posterior-{FLAGS.model}-{FLAGS.mcmc}")

    n_training_logspace = [3192]

    if FLAGS.run:
        cov_fun = get_covariance_function()

        times = np.empty(len(n_training_logspace), dtype=float)

        for i, n_training in tqdm.tqdm(enumerate(n_training_logspace), total=len(n_training_logspace)):
            t, y = get_data(n_training)
            gp_model = get_model(ModelEnum(FLAGS.model), (t, y), FLAGS.noise_variance, cov_fun,
                                 t.shape[0])
            set_gp_priors(gp_model)

            run_time, params_res = run_one_mcmc(n_training, gp_model)
            times[i] = run_time
            np.savez(f_posterior + f"-{n_training}", **params_res)
        np.save(f_times, np.stack([n_training_logspace, times], axis=1))


def main(_):
    device = tf.device(FLAGS.device)
    with device:
        run()


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs('results')
    app.run(main)
