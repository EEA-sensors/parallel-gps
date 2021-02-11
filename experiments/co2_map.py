import enum
from typing import Tuple

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from absl import app, flags
from gpflow.base import PriorOn
from gpflow.kernels import SquaredExponential
from gpflow.utilities import print_summary
from tensorflow_probability.python.distributions import Normal
from tensorflow_probability.python.experimental.mcmc import ProgressBarReducer

from pssgp.kernels import SDEPeriodic
from pssgp.kernels.matern import Matern32
from pssgp.model import StateSpaceGP

gpf.config.set_default_float(np.float32)


class ModelEnum(enum.Enum):
    GP = "GP"
    SSGP = "SSGP"
    PSSGP = "PSSGP"


class InferenceMethodEnum(enum.Enum):
    HMC = "HMC"
    MAP = "MAP"


FLAGS = flags.FLAGS

flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
flags.DEFINE_string('inference_method', InferenceMethodEnum.HMC.value, 'How to learn hyperparameters. MAP or HMC.')
flags.DEFINE_integer('periodic_order', 3, 'Order of ss-periodic approximation.', lower_bound=1)
flags.DEFINE_integer('burnin', 50, 'Burnin samples for the MCMC.', lower_bound=1)
flags.DEFINE_integer('n_samples', 50, 'Number of samples required for the MCMC.', lower_bound=1)
flags.DEFINE_float('co2_split_time', 2013.7247, 'Time to split the training and validation CO2 data.', lower_bound=1)
flags.DEFINE_boolean('plot', True, 'Plot the results. Flag it to False in Triton.')

gp_dtype = gpf.config.default_float()


def load_co2_data(split_time: float = 2015.) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Load CO2 dataset
    """
    weekly = np.loadtxt('co2_weekly_mlo.txt')[:, 3:5]
    monthly = np.loadtxt('co2_mm_mlo.txt')[:, 2:4]
    data = np.concatenate([weekly, monthly], axis=0).astype(gp_dtype)

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
    num_burnin_steps = FLAGS.burnin
    num_samples = FLAGS.n_samples
    print_summary(model)
    hmc_helper = gpf.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )
    hmc = tfp.python.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01
    )
    pbar = ProgressBarReducer(num_samples + num_burnin_steps)
    pbar.initialize(None)
    hmc = tfp.experimental.mcmc.WithReductions(hmc, pbar)

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc, num_adaptation_steps=10, target_accept_prob=0.25, adaptation_rate=0.01
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
            # trace_fn=lambda _, pkr: pkr.inner_results.inner_kernel.is_accepted,
        )

    result = run_chain_fn()
    print(result)
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)

    return dict(zip(gpf.utilities.parameter_dict(model), parameter_samples))


def map(model):
    opt = gpf.optimizers.Scipy()

    with tf.GradientTape() as tape:
        val = model.maximum_log_likelihood_objective()
    _ = tape.gradient(val, model.trainable_variables)
    print_summary(model)
    opt_logs = opt.minimize(model.training_loss, model.trainable_variables,
                            options=dict(maxiter=100, disp=True))
    print_summary(model)
    return opt_logs


def _make_cov():
    m32_cov = Matern32(variance=1000.,
                       lengthscales=100.)
    m32_cov.lengthscales.prior = Normal(100., 1000.)
    m32_cov.lengthscales.prior_on = PriorOn.UNCONSTRAINED

    m32_cov.variance.prior = Normal(100., 100.)
    m32_cov.variance.prior_on = PriorOn.UNCONSTRAINED

    periodic_base_cov = SquaredExponential(variance=1.,
                                           lengthscales=5.)

    periodic_base_cov.lengthscales.prior = Normal(1., 10.)
    periodic_base_cov.lengthscales.prior_on = PriorOn.UNCONSTRAINED

    periodic_base_cov.variance.prior = Normal(5., 10.)
    periodic_base_cov.variance.prior_on = PriorOn.UNCONSTRAINED

    periodic_cov = SDEPeriodic(periodic_base_cov,
                               period=1., order=FLAGS.periodic_order)

    periodic_cov.period.prior = Normal(1., 5.)
    periodic_cov.period.prior_on = PriorOn.UNCONSTRAINED

    periodic_damping_cov = Matern32(variance=1.,
                                    lengthscales=140.)

    periodic_damping_cov.variance.prior = Normal(5., 10.)
    periodic_damping_cov.variance.prior_on = PriorOn.UNCONSTRAINED

    periodic_damping_cov.lengthscales.prior = Normal(140., 140.)
    periodic_damping_cov.lengthscales.prior_on = PriorOn.UNCONSTRAINED

    co2_cov = periodic_cov * periodic_damping_cov + m32_cov
    return co2_cov


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
    noise_variance = 0.1

    # Prepare model
    co2_cov = _make_cov()

    def make_model():
        if model_name == ModelEnum.GP:
            model = gpf.models.GPR(data=(train_t, train_y),
                                   kernel=co2_cov,
                                   noise_variance=noise_variance,
                                   mean_function=None)
            model.likelihood.variance.prior = Normal(noise_variance, 5.)
            model.likelihood.variance.prior_on = PriorOn.UNCONSTRAINED
            return model
        if model_name == ModelEnum.SSGP:
            parallel = False
        elif model_name == ModelEnum.PSSGP:
            parallel = True
        else:
            raise ValueError(f"model_name must be a ModelEnum, found'{type(model_name)}'")
        model = StateSpaceGP(data=(train_t, train_y),
                             kernel=co2_cov,
                             noise_variance=noise_variance,
                             parallel=parallel,
                             max_parallel=4000)
        model.noise_variance.prior = Normal(noise_variance, 1.)
        model.noise_variance.prior_on = PriorOn.UNCONSTRAINED
        return model

    model = make_model()
    if inference_method == InferenceMethodEnum.MAP:
        params = map(model)
        print(params)
    elif inference_method == InferenceMethodEnum.HMC:
        samples = hmc(model)
        print(samples)
        _ = 1 + 1
    else:
        ValueError()
    #
    # if FLAGS.plot:
    #     plt.scatter(t0 + train_t, train_y, s=1)
    #     plt.scatter(val_t + t0, val_y, label='True', s=1)
    #     plt.plot(val_t + t0, m, label='predicted')
    #     plt.fill_between(val_t + t0,
    #                      m - 1.96 * np.sqrt(cov),
    #                      m + 1.96 * np.sqrt(cov),
    #                      alpha=0.2,
    #                      label='predicted')
    #
    #     plt.legend()
    #     plt.show()


if __name__ == '__main__':
    app.run(run)
