import enum

import numpy as np
from absl import flags
from gpflow import default_float

from pssgp.experiments.common import ModelEnum, CovarianceEnum
from pssgp.toymodels import sinu, comp_sinu, rect, obs_noise


class DataEnum(enum.Enum):
    SINE = "SINE"
    COMPOSITE_SINE = "COMPOSITE_SINE"
    RECT = "RECT"


FLAGS = flags.FLAGS
flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
flags.DEFINE_string('cov', CovarianceEnum.Matern32.value, 'Covariance function.')
flags.DEFINE_string('data_model', DataEnum.SINE.value, 'What is the model for the data.')
flags.DEFINE_string('dtype', "float32", 'GPFLOW default float type.')
flags.DEFINE_integer('rbf_order', 6, 'Order of ss-RBF approximation.', lower_bound=1)
flags.DEFINE_integer('rbf_balance_iter', 10, 'Iterations of RBF balancing.', lower_bound=1)
flags.DEFINE_integer('qp_order', 6, 'Order of ss-quasiperiodic approximation.', lower_bound=1)
flags.DEFINE_float('noise_variance', 0.5, 'Variance of the noise.', lower_bound=1e-4)


def get_data(seed, n_training, n_pred):
    dtype = default_float()

    t = np.linspace(0, 4, n_training, dtype=dtype)
    t_pred = np.linspace(0, 4, n_pred, dtype=dtype)
    data_model = DataEnum(FLAGS.data_model)
    if data_model == DataEnum.SINE:
        data_fun = sinu
    elif data_model == DataEnum.COMPOSITE_SINE:
        data_fun = comp_sinu
    elif data_model == DataEnum.RECT:
        data_fun = rect
    else:
        raise ValueError("")
    ft = data_fun(t)
    ft_pred = data_fun(t_pred)

    y = obs_noise(ft, FLAGS.noise_variance, seed)
    return t.reshape(-1, 1), ft.reshape(-1, 1), t_pred.reshape(-1, 1), ft_pred.reshape(-1, 1), y.reshape(-1, 1)
