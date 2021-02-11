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
from pssgp.misc_utils import rmse, error_shade

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

flags.DEFINE_integer('Nm', 400, 'Number of measurements.')
flags.DEFINE_integer('Np', 500, 'Number of predictions.')
flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
flags.DEFINE_string('cov', CovFuncEnum.Matern32.value, 'Covariance function.')
flags.DEFINE_string('inference_method', InferenceMethodEnum.MAP.value, 'How to learn hyperparameters. MAP or HMC.')
flags.DEFINE_integer('n_samples', 100, 'Number of HMC samples')
flags.DEFINE_integer('burnin', 100, 'Burning-in steps of HMC')
flags.DEFINE_integer('rbf_order', 6, 'Order of ss-RBF approximation.', lower_bound=1)
flags.DEFINE_integer('rbf_balance_iter', 10, 'Iterations of RBF balancing.', lower_bound=1)
flags.DEFINE_boolean('plot', False, 'Plot the results. Flag it to False in Triton.')

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
                            method='L-BFGS-B', options=dict(maxiter=100, disp=True))
    print_summary(model)


def run(argv):
    np.random.seed(2021)
    tf.random.set_seed(2021)
    Nm, Np = FLAGS.Nm, FLAGS.Np
    model_name = ModelEnum(FLAGS.model)
    cov_name = CovFuncEnum(FLAGS.cov)
    inference_method = InferenceMethodEnum(FLAGS.inference_method)

    # Generate data
    t = np.linspace(0, 4, Nm)
    ft = sinu(t)
    y = obs_noise(ft, 0.1)
    data = (tf.constant(t[:, None], dtype=gp_dtype),
            tf.constant(y[:, None], dtype=gp_dtype))

    t_pred = np.linspace(0, 4, Np).reshape(-1, 1)
    ft_pred = sinu(t_pred)

    # Prepare cov
    if cov_name == CovFuncEnum.Matern32:
        cov_func = Matern32(variance=1.,
                            lengthscales=0.5)

    elif cov_name == CovFuncEnum.RBF:
        cov_func = RBF(variance=1.,
                       lengthscales=0.5,
                       order=FLAGS.rbf_order, balancing_iter=FLAGS.rbf_balance_iter)
    else:
        raise ValueError('Covariance function not found')
    cov_func.lengthscales.prior = Normal(f64(0.5), f64(2))
    cov_func.variance.prior = Normal(f64(1.), f64(2.))

    if model_name == ModelEnum.GP:
        model = gpf.models.GPR(data=data,
                               kernel=cov_func,
                               noise_variance=1.,
                               mean_function=None)
        model.likelihood.variance.prior = Normal(f64(0.5), f64(1.))
    elif model_name == ModelEnum.SSGP:
        model = StateSpaceGP(data=data,
                             kernel=cov_func,
                             noise_variance=1.,
                             parallel=False)
        model._noise_variance.prior = Normal(f64(0.5), f64(1.))
    elif model_name == ModelEnum.PSSGP:
        model = StateSpaceGP(data=data,
                             kernel=cov_func,
                             noise_variance=1.,
                             parallel=True,
                             max_parallel=4000)
        model._noise_variance.prior = Normal(f64(0.5), f64(1.))
    else:
        raise ValueError('Model {} not found.'.format(FLAGS.model))

    print_summary(model)

    print('>>>>>>>Learning hyperparameters.')
    if inference_method == InferenceMethodEnum.MAP:
        map(model)
    elif inference_method == InferenceMethodEnum.HMC:
        hmc(model)
    else:
        ValueError('Method {} not found.'.format(inference_method))

    print('>>>>>>>Making predictions.')
    m, cov = model.predict_f(t_pred)
    print('>>>>>>>RMSE: {}.'.format(rmse(m, ft_pred)))

    filename = 'results/sinu_{}_{}_{}_{}_{}_{}_{}_{}_{}.npz'.format(FLAGS.Nm,
                                                                    FLAGS.Np,
                                                                    FLAGS.model,
                                                                    FLAGS.cov,
                                                                    FLAGS.inference_method,
                                                                    FLAGS.n_samples,
                                                                    FLAGS.burnin,
                                                                    FLAGS.rbf_order,
                                                                    FLAGS.rbf_balance_iter)
    np.savez(filename, m, cov, t, y, t_pred, rmse(m, ft_pred))
    print('>>>>>>>File save to {}'.format(filename))

    if FLAGS.plot:
        plt.scatter(t, y, label='Measurements', s=1)
        plt.plot(t, ft, label='True')
        plt.plot(t_pred, m, label='predicted')
        error_shade(t_pred, m, cov, alpha=0.2, label='.95 conf')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    app.run(run)
