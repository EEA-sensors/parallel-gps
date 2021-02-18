import enum
import time

import gpflow as gpf
import numpy as np
import tensorflow as tf
from absl import flags
from absl.flags import FLAGS
from gpflow import config
from gpflow.kernels import SquaredExponential
from gpflow.models import GPR
from tensorflow_probability.python.experimental.mcmc import ProgressBarReducer, WithReductions, \
    make_tqdm_progress_bar_fn
from tensorflow_probability.python.mcmc import HamiltonianMonteCarlo, MetropolisAdjustedLangevinAlgorithm, \
    NoUTurnSampler, sample_chain

from pssgp.kernels import Matern12, Matern32, Matern52, RBF, Periodic
from pssgp.model import StateSpaceGP


class MCMC(enum.Enum):
    HMC = "HMC"
    MALA = "MALA"
    NUTS = "NUTS"


class ModelEnum(enum.Enum):
    GP = "GP"
    SSGP = "SSGP"
    PSSGP = "PSSGP"


class CovarianceEnum(enum.Enum):
    Matern12 = 'Matern12'
    Matern32 = 'Matern32'
    Matern52 = 'Matern52'
    RBF = "RBF"
    QP = "QP"


flags.DEFINE_string("device", "/cpu:0", "Device on which to run")


def get_simple_covariance_function(covariance_enum, **kwargs):
    if not isinstance(covariance_enum, CovarianceEnum):
        covariance_enum = CovarianceEnum(covariance_enum)
    if covariance_enum == CovarianceEnum.Matern12:
        return Matern12(**kwargs)
    if covariance_enum == CovarianceEnum.Matern32:
        return Matern32(**kwargs)
    if covariance_enum == CovarianceEnum.Matern52:
        return Matern52(**kwargs)
    if covariance_enum == CovarianceEnum.RBF:
        return RBF(**kwargs)
    if covariance_enum == CovarianceEnum.QP:
        base_kernel = SquaredExponential(kwargs.pop("variance", 1.), kwargs.pop("lengthscales", 1.))
        return Periodic(base_kernel, **kwargs)


def get_model(model_enum, data, noise_variance, covariance_function, max_parallel=10000):
    if not isinstance(model_enum, ModelEnum):
        model_enum = ModelEnum(model_enum)
    if model_enum == ModelEnum.GP:
        gp_model = GPR(data, covariance_function, None, noise_variance)
    elif model_enum == ModelEnum.SSGP:
        gp_model = StateSpaceGP(data, covariance_function, noise_variance, parallel=False)
    elif model_enum == ModelEnum.PSSGP:
        gp_model = StateSpaceGP(data, covariance_function, noise_variance, parallel=True, max_parallel=max_parallel)
    else:
        raise ValueError("model not supported")
    return gp_model


def run_one_mcmc(n_training, gp_model):
    num_burnin_steps = FLAGS.n_burnin
    num_samples = FLAGS.n_samples

    mcmc_helper, run_chain_fn = get_run_chain_fn(gp_model, num_samples, num_burnin_steps)
    try:
        tic = time.time()
        result, is_accepted = run_chain_fn()
        print(np.mean(is_accepted))
        run_time = time.time() - tic
        parameter_samples = mcmc_helper.convert_to_constrained_values(result)

    except Exception as e:  # noqa: It's not clear what the error returned by TF could be, so well...
        run_time = float("nan")
        parameter_samples = [np.nan * np.ones((num_samples,), dtype=config.default_float()) for _ in
                             gp_model.trainable_parameters]
        print(f"{FLAGS.model}-{FLAGS.cov} failed with n_training={n_training} and error: \n {e}")

    return run_time, dict(zip(gpf.utilities.parameter_dict(gp_model), parameter_samples))


def get_run_chain_fn(gp_model, num_samples, num_burnin_steps):
    mcmc_helper = gpf.optimizers.SamplingHelper(
        gp_model.log_posterior_density, gp_model.trainable_parameters)

    if FLAGS.mcmc == MCMC.HMC.value:
        mcmc = HamiltonianMonteCarlo(
            target_log_prob_fn=mcmc_helper.target_log_prob_fn,
            num_leapfrog_steps=FLAGS.n_leapfrogs,
            step_size=FLAGS.step_size
        )
    elif FLAGS.mcmc == MCMC.MALA.value:
        mcmc = MetropolisAdjustedLangevinAlgorithm(
            target_log_prob_fn=mcmc_helper.target_log_prob_fn,
            step_size=FLAGS.step_size
        )
    elif FLAGS.mcmc == MCMC.NUTS.value:
        mcmc = NoUTurnSampler(
            target_log_prob_fn=mcmc_helper.target_log_prob_fn,
            step_size=FLAGS.step_size
        )
    else:
        raise ValueError(f"mcmc must be a {MCMC} enum, {FLAGS.mcmc} was passed")
    pbar = ProgressBarReducer(num_samples + num_burnin_steps,
                              make_tqdm_progress_bar_fn(f"{FLAGS.model}-{FLAGS.mcmc}", True))
    pbar.initialize(None)

    mcmc = WithReductions(mcmc, pbar)

    @tf.function
    def run_chain_fn():
        return sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=mcmc_helper.current_state,
            kernel=mcmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
        )

    return mcmc_helper, run_chain_fn
