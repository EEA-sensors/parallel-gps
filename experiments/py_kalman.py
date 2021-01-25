"""
Comparison with Pykalman

Note that the PyKalman package does not seem to support parameter estimation.
See, https://pykalman.github.io/#optimizing-parameters, they said
'... It is customary optimize only the transition_covariance, observation_covariance, initial_state_mean,
and initial_state_covariance, ...'

To compare, we perhaps need to manually put parameters here.

"""
import matplotlib.pylab as plt
import numpy as np
from pykalman import KalmanFilter
from scipy.linalg import expm

from src.kalman.base import LGSSM
from src.kalman.sequential import kf as our_kf, kfs as ourkfs
from src.toymodels import sinu, obs_noise

# Generate data
np.random.seed(666)

N = 500
t = np.linspace(0, 1, N)
dt = np.diff(t)[0]
ft = sinu(t)
y = obs_noise(ft, 0.01 * np.eye(t.shape[0]))
# y[50:100] = np.nan
# GP paras
lengthscale = 0.1
sigma = 1.

# Get disc state-space model
cov_func = 'matern32'
lam = 1 / lengthscale

if cov_func == 'matern12':
    F = np.array([-lam]).reshape(1, 1)
    L = np.array(1.).reshape(1, 1)
    D = 1
    H = np.array(1.).reshape(1, 1)
    m0 = np.array(0.).reshape(1, 1)
    P0 = np.array(sigma ** 2).reshape(1, 1)

elif cov_func == 'matern32':
    F = np.array([[0., 1],
                  [-lam ** 2, -2 * lam]])
    L = np.array([0., 1.]).reshape(2, 1)
    D = 2
    H = np.array([1., 0.]).reshape(1, 2)
    m0 = np.zeros((2,))
    P0 = np.array([[sigma ** 2, 0.],
                   [0, lam ** 2 * sigma ** 2]])

elif cov_func == 'matern52':
    F = np.array([[0., 1, 0],
                  [0, 0, 1],
                  [-lam ** 3, -3 * (lam ** 2), -3 * lam]])
    L = np.array([0., 0., 1.]).reshape(3, 1)
    D = 3
    H = np.array([1., 0., 0.]).reshape(1, 3)
    m0 = np.zeros((3,))
    P0 = np.array([[sigma ** 2, 0, -1 / 3 * lam ** 2 * sigma ** 2],
                   [0, 1 / 3 * lam ** 2 * sigma ** 2, 0],
                   [-1 / 3 * lam ** 2 * sigma ** 2, 0, lam ** 4 * sigma ** 2]])

q = sigma ** 2 * (np.math.factorial(D - 1) ** 2 / np.math.factorial(2 * D - 2)) * (2 * lam) ** (2 * D - 1)
A = expm(F * dt)
n = F.shape[0]
Phi = np.concatenate([np.concatenate([F, L @ np.transpose(L) * q], axis=1),
                      np.concatenate([np.zeros((n, n)), -np.transpose(F)], axis=1)],
                     axis=0)
AB = expm(Phi * dt) @ np.concatenate([np.zeros((n, n)), np.eye(n)], axis=0)
Q = AB[:n, :] @ np.transpose(A)
# Q = np.transpose(np.linalg.solve(np.transpose(AB[n:, :]), np.transpose(AB[:n, :])))


kf = KalmanFilter(transition_matrices=A,
                  transition_covariance=Q,
                  observation_matrices=H,
                  initial_state_mean=m0,
                  initial_state_covariance=P0,
                  n_dim_obs=1)

(filtered_state_means, filtered_state_covariances) = kf.filter(y)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(y)

lgssm = LGSSM(P0,
              np.repeat(np.expand_dims(A, 0), N, axis=0),
              np.repeat(np.expand_dims(Q, 0), N, axis=0),
              H,
              np.eye(1))

our_filtered = our_kf(lgssm, y.reshape((-1, 1)))
our_smoothed = ourkfs(lgssm, y.reshape((-1, 1)))
# plt.plot(t, ft, c='r')
# plt.plot(t, our_filtered[0][:, 0] - filtered_state_means[:, 0], c='b')
plt.plot(t, our_smoothed[0][:, 1], c='orange')
# plt.plot(t, filtered_state_means[:, 0], c='g')
# plt.plot(t, smoothed_state_means[:, 1], c='k')
plt.show()
