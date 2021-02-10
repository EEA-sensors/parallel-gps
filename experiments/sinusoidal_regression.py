# Regression experiments on sinusoidal signals.
# Corresponds to the *** of paper.

import timeit
import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

from absl import app, flags
from typing import Tuple

from gpflow.kernels import SquaredExponential
from gpflow.utilities import print_summary
from gpflow.ci_utils import ci_niter
from tensorflow_probability.python.distributions import Normal

from pssgp.kernels import RBF, Matern32
from pssgp.model import StateSpaceGP
from pssgp.toymodels import sinu, obs_noise

import enum


class ModelEnum(enum.Enum):
    GP = "GP"
    SSGP = "SSGP"
    PSSGP = "PSSGP"


class InferenceMethodEnum(enum.Enum):
    HMC = "HMC"
    MAP = "MAP"


class CovFuncEnum(enum.Enum):
    Matern32 = 'Matern32'
    RBF = "RBF"


FLAGS = flags.FLAGS

flags.DEFINE_integer('N', 1000, 'Number of measurements.')
flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
flags.DEFINE_string('cov', CovFuncEnum.Matern32.value, 'Covariance function.')
flags.DEFINE_string('inference_method', InferenceMethodEnum.MAP.value, 'How to learn hyperparameters. MAP or HMC.')
flags.DEFINE_integer('n_samples', 100, 'Number of HMC samples')
flags.DEFINE_integer('burnin', 100, 'Burning-in steps of HMC')
flags.DEFINE_integer('rbf_order', 6, 'Order of ss-RBF approximation.', lower_bound=1)
flags.DEFINE_integer('rbf_balance_iter', 10, 'Iterations of RBF balancing.', lower_bound=1)
flags.DEFINE_boolean('plot', True, 'Plot the results. Flag it to False in Triton.')

gp_dtype = gpf.config.default_float()
f64 = gpf.utilities.to_default_float


def hmc(model):
    # :ref:`https://gpflow.readthedocs.io/en/master/notebooks/advanced/mcmc.html`
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
    N = FLAGS.N
    model_name = ModelEnum(FLAGS.model)
    cov_name = CovFuncEnum(FLAGS.cov)
    inference_method = InferenceMethodEnum(FLAGS.inference_method)

    # Load data
    t = np.linspace(-2, 2, N)
    ft = sinu(t)
    y = obs_noise(ft, 0.1)
    data = (tf.constant(t[:, None], dtype=gp_dtype),
            tf.constant(y[:, None], dtype=gp_dtype))
    # TODO: Number of query points?

    # Prepare cov
    if cov_name == CovFuncEnum.Matern32:
        cov_func = Matern32(variance=1.,
                            lengthscales=0.5)
        cov_func.lengthscales.prior = Normal(0.5, 2)
        cov_func.variance.prior = Normal(1, 2)
    elif cov_name == CovFuncEnum.Matern32:
        cov_func = RBF(variance=1.,
                       lengthscales=0.5,
                       order=FLAGS.rbf_order, balancing_iter=FLAGS.rbf_balance_iter)
        cov_func.lengthscales.prior = Normal(0.5, 2)
        cov_func.variance.prior = Normal(1, 2)
    else:
        raise ValueError('Covariance function not found')

    if model_name == ModelEnum.GP:
        model = gpf.models.GPR(data=data,
                               kernel=cov_func,
                               noise_variance=1.,
                               mean_function=None)
    elif model_name == ModelEnum.SSGP:
        model = StateSpaceGP(data=data,
                             kernel=cov_func,
                             noise_variance=1.,
                             parallel=False,
                             max_parallel=4000)
    elif model_name == ModelEnum.PSSGP:
        model = StateSpaceGP(data=data,
                             kernel=cov_func,
                             noise_variance=1.,
                             parallel=True,
                             max_parallel=4000)
    else:
        raise ValueError('Model {} not found.'.format(FLAGS.model))

    print_summary(model)

    print('>>>>>>>Learning hyperparameters.')
    if inference_method == InferenceMethodEnum.MAP:
        map(model)
    elif inference_method == InferenceMethodEnum.HMC:
        hmc(model)
    else:
        ValueError('Method not found.')

    print('>>>>>>>Making predictions.')
    m, cov = model.predict_f(val_t)

    filename = 'results/sinu_{}_{}_{}_{}_{}_{}_{}_{}.npz'.format(FLAGS.N,
                                                                 FLAGS.model,
                                                                 FLAGS.cov,
                                                                 FLAGS.inference_method,
                                                                 FLAGS.n_samples,
                                                                 FLAGS.burnin,
                                                                 FLAGS.rbf_order,
                                                                 FLAGS.rbf_balance_iter)
    np.savez(filename, m, cov)
    print('>>>>>>>File save to {}'.format(filename))

    if FLAGS.plot:
        plt.scatter(t, y, label='Measurements', s=1)
        plt.plot(t, ft, label='True')
        plt.plot(t, m, label='predicted')
        plt.fill_between(t,
                         m - 1.96 * np.sqrt(cov),
                         m + 1.96 * np.sqrt(cov),
                         alpha=0.2)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    app.run(run)
