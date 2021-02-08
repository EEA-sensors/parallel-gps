import timeit
import argparse
import gpflow as gpf
import numpy as np
import numpy.testing as npt
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from absl import app, flags
from typing import Tuple

from gpflow.kernels import SquaredExponential
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
from tensorflow_probability.python.distributions import Normal

from pssgp.kernels import RBF, SDEPeriodic
from pssgp.kernels.matern import Matern32
from pssgp.model import StateSpaceGP

import enum


class ModelEnum(enum.Enum):
    GP = "GP"
    SSGP = "SSGP"
    PSSGP = "PSSGP"


class InferenceMethodEnum(enum.Enum):
    HMC = "HMC"
    MAP = "MAP"


FLAGS = flags.FLAGS

flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
flags.DEFINE_string('inference_method', InferenceMethodEnum.MAP.value, 'How to learn hyperparameters. MAP or HMC.')
flags.DEFINE_integer('rbf_order', 6, 'Order of ss-RBF approximation.', lower_bound=1)
flags.DEFINE_integer('rbf_balance_iter', 10, 'Iterations of RBF balancing.', lower_bound=1)
flags.DEFINE_integer('periodic_order', 6, 'Order of ss-periodic approximation.', lower_bound=1)
flags.DEFINE_float('co2_split_time', 2013.7247, 'Time to split the training and validation CO2 data.', lower_bound=1)
flags.DEFINE_boolean('plot', True, 'Plot the results. Flag it to False in Triton.')

gp_dtype = gpf.config.default_float()
f64 = gpf.utilities.to_default_float


def load_co2_data(split_time: float = 2015.) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load CO2 dataset
    """
    weekly = np.loadtxt('co2_weekly_mlo.txt')[:, 3:5]
    monthly = np.loadtxt('co2_mm_mlo.txt')[:, 2:4]
    data = np.concatenate([weekly, monthly], axis=0)

    # Remove invalid data -999.99 in co2 column
    rm_mask = np.any(data < 0, axis=1)
    data = data[~rm_mask]

    # Sort data in temporal order
    idx = np.argsort(data, axis=0)
    idx[:, 1] = idx[:, 0]
    data = tf.constant(np.take_along_axis(data, idx, axis=0), dtype=gp_dtype)

    # Split training and validation data
    train_data = data[data[:, 0] <= split_time]
    val_data = data[data[:, 0] > split_time]

    # Return t, y, t, y in tf.Tensor
    return train_data[:, 0, None], train_data[:, 1, None], val_data[:, 0, None], val_data[:, 1, None]


def hmc(model):
    # https://gpflow.readthedocs.io/en/master/notebooks/advanced/mcmc.html
    num_burnin_steps = ci_niter(300)
    num_samples = ci_niter(500)

    hmc_helper = gpf.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
    )
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=10, target_accept_prob=f64(0.75), adaptation_rate=0.1
    )

    with tf.GradientTape() as tape:
        val = model.maximum_log_likelihood_objective()
    _ = tape.gradient(val, model.trainable_variables)

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )

    samples, traces = run_chain_fn()
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)
    print(parameter_samples)

    param_to_name = {param: name for name, param in gpf.utilities.parameter_dict(model).items()}


def map(model):
    opt = gpf.optimizers.Scipy()

    with tf.GradientTape() as tape:
        val = model.maximum_log_likelihood_objective()
    _ = tape.gradient(val, model.trainable_variables)

    opt_logs = opt.minimize(model.training_loss, model.trainable_variables,
                            options=dict(maxiter=100, disp=True))
    print_summary(model)


def run(argv):
    np.random.seed(2021)
    tf.random.set_seed(2021)
    model_name = ModelEnum(FLAGS.model)
    inference_method = InferenceMethodEnum(FLAGS.inference_method)

    # Load data
    train_t, train_y, val_t, val_y = load_co2_data(FLAGS.co2_split_time)
    t0 = tf.reduce_min(train_t)
    train_t = train_t - t0
    val_t = val_t - t0
    y_mean = tf.reduce_mean(tf.concat([train_y, val_y], axis=0))
    train_y = train_y - y_mean
    val_y = val_y - y_mean
    noise_variance = 1.

    # Prepare model
    m32_cov = Matern32(variance=Normal(f64(1.), f64(2.)),
                       lengthscales=Normal(f64(0.5), f64(2.)))
    rbf_cov = RBF(variance=Normal(f64(100.), f64(100.)),
                  lengthscales=Normal(f64(10.), f64(20.)),
                  order=FLAGS.rbf_order, balancing_iter=FLAGS.rbf_balance_iter)
    periodic_base_cov = SquaredExponential(variance=Normal(f64(1.), f64(2.)),
                                           lengthscales=Normal(f64(5.), f64(5.)))
    periodic_cov = SDEPeriodic(periodic_base_cov,
                               period=Normal(f64(1.), f64(2.)), order=FLAGS.periodic_order)
    periodic_damping_cov = Matern32(variance=Normal(f64(1.), f64(2.)),
                                    lengthscales=Normal(f64(140.), f64(140.)))

    co2_cov = rbf_cov + periodic_cov * periodic_damping_cov + m32_cov
    # co2_cov = m32_cov + periodic_cov

    def run_gp():
        model = gpf.models.GPR(data=(train_t, train_y),
                               kernel=co2_cov,
                               noise_variance=noise_variance,
                               mean_function=None)
        if inference_method == InferenceMethodEnum.MAP:
            map(model)
        elif inference_method == InferenceMethodEnum.HMC:
            hmc(model)
        else:
            ValueError('Method not found.')

        m, cov = model.predict_f(val_t)
        return m, cov

    def run_pssgp(parallel):
        model = StateSpaceGP(data=(train_t, train_y),
                             kernel=co2_cov,
                             noise_variance=noise_variance,
                             parallel=parallel,
                             max_parallel=3175)
        print_summary(model)
        if inference_method == InferenceMethodEnum.MAP:
            map(model)
        elif inference_method == InferenceMethodEnum.HMC:
            hmc(model)
        else:
            ValueError('Method not found.')

        m, cov = model.predict_f(val_t)
        return m, cov

    if model_name == ModelEnum.GP:
        m, cov= run_gp()
    elif model_name == ModelEnum.SSGP:
        m, cov = run_pssgp(False)
    elif model_name == ModelEnum.PSSGP:
        m, cov= run_pssgp(True)
    else:
        ValueError('Model {} does not exist.'.format(FLAGS.model))

    print(cov)
    if FLAGS.plot:
        plt.scatter(t0 + train_t, train_y, s=1)
        plt.scatter(val_t + t0, val_y, label='True', s=1)
        plt.plot(val_t + t0, m, label='predicted')
        plt.fill_between(val_t + t0,
                         m - 1.96 * np.sqrt(cov),
                         m + 1.96 * np.sqrt(cov),
                         alpha=0.2,
                         label='predicted')

        plt.legend()
        plt.show()


if __name__ == '__main__':
    app.run(run)
