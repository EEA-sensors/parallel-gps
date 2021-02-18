import enum
import os

import numpy as np
import tensorflow as tf
from absl import flags
from gpflow import default_float

from pssgp.experiments.common import ModelEnum


class DataEnum(enum.Enum):
    SINE = "SINE"
    COMPOSITE_SINE = "COMPOSITE_SINE"
    RECT = "RECT"


FLAGS = flags.FLAGS
flags.DEFINE_string('model', ModelEnum.SSGP.value, 'Select model to run. Options are gp, ssgp, and pssgp.')
flags.DEFINE_string('data_model', DataEnum.SINE.value, 'What is the model for the data.')
flags.DEFINE_string('data_dir', "", 'Directory of the data.')
flags.DEFINE_string('dtype', "float64", 'GPFLOW default float type.')
flags.DEFINE_integer('qp_order', 3, 'Order of ss-quasiperiodic approximation.', lower_bound=1)
flags.DEFINE_float('noise_variance', 0.05, 'Variance of the noise.', lower_bound=1e-4)


# TODO: Put a flag for results dumping.



def get_data(n_training):
    dtype = default_float()

    weekly = np.loadtxt(os.path.join(FLAGS.data_dir, 'co2_weekly_mlo.txt'))[:, 3:5]
    monthly = np.loadtxt(os.path.join(FLAGS.data_dir, 'co2_mm_mlo.txt'))[:, 2:4]
    data = np.concatenate([weekly, monthly], axis=0).astype(dtype)

    # Remove invalid data -999.99 in co2 column
    rm_mask = np.any(data < 0, axis=1)
    data = data[~rm_mask]

    # Sort data in temporal order
    idx = np.argsort(data, axis=0)
    idx[:, 1] = idx[:, 0]
    data = tf.constant(np.take_along_axis(data, idx, axis=0), dtype=dtype)

    # Split training and validation data
    train_data = data[-n_training:]

    # Return t, y, t, y in tf.Tensor
    return train_data[:, 0, None], train_data[:, 1, None]
