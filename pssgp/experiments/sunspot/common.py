import enum
import os

import numpy as np
import pandas as pd
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
flags.DEFINE_float('noise_variance', 10., 'Variance of the noise.', lower_bound=1e-4)


# TODO: Put a flag for results dumping.

def get_data(n_training):
    data = pd.read_csv(os.path.join(FLAGS.data_dir, 'sunspots.csv'), parse_dates=[1], index_col=0, header=0)
    t = ((data["date"] - data.loc[0, "date"]) / np.timedelta64(1, "Y")).values
    vals = data["sunspots"].values
    return t[-n_training:, None], vals[-n_training:, None]
