# Regression experiments on sinusoidal signals.
# Corresponds to the *** of paper.
import os
import time
from itertools import product

import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags
from gpflow.models.util import data_input_to_tensor
from scipy.stats.kde import gaussian_kde

from pssgp.experiments.common import ModelEnum, CovarianceEnum, get_simple_covariance_function, get_model
from pssgp.experiments.toy_models.common import FLAGS, get_data
from pssgp.misc_utils import rmse

flags.DEFINE_integer('n_seeds', 21, "Seed for numpy random generator")
flags.DEFINE_integer('mesh_size', 10, "Size of the mesh for prediction")
flags.DEFINE_boolean('plot', False, "Plot the result")
flags.DEFINE_boolean('run', True, "Run the result or load the data")


def run_one(seed, covariance_function, gp_model, n_training, n_pred):
    t, ft, t_pred, ft_pred, y = get_data(seed, n_training, n_pred)
    gp_dtype = gpf.config.default_float()

    if gp_model is None:
        model_name = ModelEnum(FLAGS.model)
        gp_model = get_model(model_name, (t, y), FLAGS.noise_variance, covariance_function,
                             t.shape[0] + t_pred.shape[0])
    else:
        gp_model.data = data_input_to_tensor((t, y))

    tensor_t_pred = tf.convert_to_tensor(t_pred, dtype=gp_dtype)
    y_pred, _ = gp_model.predict_f(tensor_t_pred)
    error = rmse(y_pred, ft_pred)
    return error, gp_model


def ridgeline(ax, data, overlap=0, fill=True, fill_color="b", n_points=150):
    """
    Adapted from https://glowingpython.blogspot.com/2020/03/ridgeline-plots-in-pure-matplotlib.html
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    ys = []
    for i, d in enumerate(data):
        pdf = gaussian_kde(d)
        y = i * (1.0 - overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            ax.fill_between(xx, np.ones(n_points) * y,
                            curve + y, zorder=len(data) - i + 1, color=fill_color, alpha=0.5)
        ax.plot(xx, curve + y, zorder=len(data) - i + 1, color=fill_color)


def run():
    f_stability = os.path.join("results", f"stability-matrix-{FLAGS.cov}-{FLAGS.model}")
    f_time = os.path.join("results", f"time-matrix-{FLAGS.cov}-{FLAGS.model}")

    if FLAGS.run:
        gpf.config.set_default_float(getattr(np, FLAGS.dtype))
        cov_name = CovarianceEnum(FLAGS.cov)
        cov_fun = get_simple_covariance_function(cov_name)
        errors = np.empty((FLAGS.mesh_size, FLAGS.mesh_size, FLAGS.n_seeds), dtype=float)
        times = np.empty((FLAGS.mesh_size, FLAGS.mesh_size, FLAGS.n_seeds), dtype=float)
        n_training_logspace = n_test_logspace = np.logspace(12, 15, FLAGS.mesh_size, base=2, dtype=int)

        for (i, n_training), (j, n_pred) in tqdm.tqdm(product(enumerate(n_training_logspace),
                                                              enumerate(n_test_logspace)),
                                                      total=FLAGS.mesh_size ** 2,
                                                      desc=FLAGS.model):
            model = None
            for seed in tqdm.trange(FLAGS.n_seeds, leave=False):
                try:
                    tic = time.time()
                    error, model = run_one(seed, cov_fun, model, n_training, n_pred)
                    toc = time.time()
                    # the only reason we return the model is so that we don't have to recompile everytime
                    errors[i, j, seed] = error
                    times[i, j, seed] = toc - tic
                except Exception as e:  # noqa: It's not clear what the error returned by TF could be, so well...
                    errors[i, j, seed] = float("nan")
                    times[i, j, seed] = float("nan")
                    print(
                        f"{FLAGS.model}-{FLAGS.cov} failed with n_training,n_pred={n_training, n_pred} and error: \n {e}")

        np.save(f_stability, errors)
        np.save(f_time, times)
    elif FLAGS.plot:
        errors = np.load(f_stability + ".npy")
        fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(17, 12))
        for i, ax in enumerate(axes[0, :]):
            ridgeline(ax, errors[(i + 1) * FLAGS.mesh_size // 3 - 1])
        for j, ax in enumerate(axes[1, :]):
            ridgeline(ax, errors[:, (j + 1) * FLAGS.mesh_size // 3 - 1])
        fig.show()


def main(_):
    device = tf.device(FLAGS.device)
    with device:
        run()


if __name__ == '__main__':
    if not os.path.exists("results"):
        os.makedirs('results')
    app.run(main)
