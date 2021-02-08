import timeit
import argparse
import gpflow as gpf
import numpy as np
import numpy.testing as npt
import tensorflow as tf

from absl import app, flags

from gpflow.kernels import SquaredExponential
from pssgp.kernels import RBF, SDEPeriodic
from pssgp.kernels.matern import Matern32
from pssgp.model import StateSpaceGP


from .common import get_gp

# def run(T, rbf_order, periodic_order, device, model_type):
#     np_data = ...
#     with tf.device(device):
#         tf_data = not tf.conver..
#         kernel = ...
#         gp = get_gp(kernel, tf_data)
#         trainable_params = gp.trainable_parameters()
#         with tf.GradientTape() as tape:
#             _val = gp.log...
#         _ = tf.gradient(_val, trainable_params)
#
#         timeit.timeit(lambda : mini())

FLAGS = flags.FLAGS

flags.DEFINE_string('hyperpara', 'map', 'Methof for learning hyperparameters')
flags.DEFINE_integer('rbf_order', 3, 'Order of ss-RBF approximation.', lower_bound=1)
flags.DEFINE_integer('periodic_order', 6, 'Order of ss-periodic approximation.', lower_bound=1)


def load_co2_data():
    pass


def main(argv):
    np.random.seed(2021)
    tf.random.set_seed(2021)


if __name__ == '__main__':
  app.run(main)