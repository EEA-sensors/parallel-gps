# Regression experiments on sinusoidal signals.
# Corresponds to the *** of paper.
import os
import time

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from gpflow.base import PriorOn
from gpflow.models import GPModel
from tensorflow_probability.python.distributions import Normal

from pssgp.experiments.common import ModelEnum, CovarianceEnum, get_covariance_function, get_model, get_run_chain_fn, \
    MCMC
from pssgp.experiments.toy_models.common import FLAGS, get_data

flags.DEFINE_integer('np_seed', 42, "data model seed")
flags.DEFINE_integer('tf_seed', 31415, "mcmc model seed")
flags.DEFINE_integer('n_runs', 1, "size of the logspace for n training samples")
flags.DEFINE_integer('n_samples', 100, "Number of samples required")
flags.DEFINE_integer('n_burnin', 50, "Number of burnin samples")
flags.DEFINE_string('mcmc', MCMC.HMC.value, "Which MCMC algo")
flags.DEFINE_float('step_size', 0.1, "Step size for the gradient based chain")
flags.DEFINE_float('n_leapfrogs', 10, "Num leapfrogs for HMC")

flags.DEFINE_boolean('plot', False, "Plot the result")
flags.DEFINE_boolean('run', True, "Run the result or load the data")


def set_priors(gp_model: GPModel):
    to_dtype = gpf.utilities.to_default_float
    if FLAGS.model == ModelEnum.GP:
        gp_model.likelihood.variance.prior = Normal(to_dtype(0.1), to_dtype(1.))
        gp_model.likelihood.variance.prior_on = PriorOn.UNCONSTRAINED
    else:
        gp_model.noise_variance.prior = Normal(to_dtype(0.1), to_dtype(1.))
        gp_model.noise_variance.prior_on = PriorOn.UNCONSTRAINED

    gp_model.kernel.variance.prior = Normal(to_dtype(1.), to_dtype(3.))
    gp_model.kernel.variance.prior_on = PriorOn.UNCONSTRAINED

    gp_model.kernel.lengthscales.prior = Normal(to_dtype(1.), to_dtype(3.))
    gp_model.kernel.lengthscales.prior_on = PriorOn.UNCONSTRAINED
    return gp_model


def run_one(n_training):
    cov_name = CovarianceEnum(FLAGS.cov)
    num_burnin_steps = FLAGS.n_burnin
    num_samples = FLAGS.n_samples
    t, ft, *_, y = get_data(FLAGS.np_seed, n_training, 1)

    cov_fun = get_covariance_function(cov_name)

    model_name = ModelEnum(FLAGS.model)
    gp_model = get_model(model_name, (t, y), FLAGS.noise_variance, cov_fun,
                         t.shape[0])
    gp_model = set_priors(gp_model)
    mcmc_helper, run_chain_fn = get_run_chain_fn(gp_model, num_samples, num_burnin_steps)
    tic = time.time()
    result, all_traces = run_chain_fn()
    toc = time.time()

    parameter_samples = mcmc_helper.convert_to_constrained_values(result)

    return toc - tic, dict(zip(gpf.utilities.parameter_dict(gp_model), parameter_samples))


def run():
    tf.random.set_seed(FLAGS.tf_seed)
    f_times = os.path.join("results", f"mcmc-times-{FLAGS.cov}-{FLAGS.model}-{FLAGS.mcmc}")
    f_posterior = os.path.join("results", f"mcmc-posterior-{FLAGS.cov}-{FLAGS.model}-{FLAGS.mcmc}")
    if FLAGS.cov == CovarianceEnum.QP.value:
        raise NotImplementedError("Quasiperiodic is not supported for this experiment")
    n_training_logspace = np.logspace(7, 14, FLAGS.n_runs, base=2, dtype=int)

    if FLAGS.run:
        params_res = dict()
        times = np.empty(FLAGS.n_runs, dtype=float)
        gpf.config.set_default_float(getattr(np, FLAGS.dtype))
        for i, n_training in tqdm.tqdm(enumerate(n_training_logspace), total=FLAGS.n_runs):
            run_time, params_res = run_one(n_training)
            times[i] = run_time
            np.savez(f_posterior + f"-{n_training}", **params_res)
        np.save(f_times, np.stack([n_training_logspace, times], axis=1))


def main(_):
    with tf.device(FLAGS.device):
        run()


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs('results')
    app.run(main)
