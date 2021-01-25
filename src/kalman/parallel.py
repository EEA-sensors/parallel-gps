import math
from functools import partial

import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalFullCovariance
from tensorflow_probability.python.math import scan_associative

__all__ = ["pkf", "pks", "pkfs"]

mv = tf.linalg.matvec
mm = tf.linalg.matmul


@partial(tf.function, experimental_relax_shapes=True)
def first_filtering_element(m0, P0, F, Q, H, R, y):
    m1 = mv(F, m0)
    P1 = F @ mm(P0, F, transpose_b=True) + Q

    def _res_nan():
        A = F
        b = m1
        C = P1
        eta = tf.zeros_like(y)
        J = tf.zeros((y.shape[0], y.shape[0]), dtype=y.dtype)
        return A, b, C, J, eta

    def _res_not_nan():
        S1 = H @ mm(P1, H, transpose_b=True) + R
        S1_chol = tf.linalg.cholesky(S1)
        K1t = tf.linalg.cholesky_solve(S1_chol, H @ P1)

        A = tf.zeros_like(F)
        b = m1 + mv(K1t, y - mv(H, m1), transpose_a=True)
        C = P1 - mm(K1t, S1, transpose_a=True) @ K1t

        S = H @ mm(Q, H, transpose_b=True) + R
        chol = tf.linalg.cholesky(S)
        HF = H @ F
        eta = mv(HF,
                 tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y, 1)), 1),
                 transpose_a=True)
        J = mm(HF, tf.linalg.cholesky_solve(chol, H @ F), transpose_a=True)
        return A, b, C, J, eta

    return tf.cond(tf.math.is_nan(y), _res_nan, _res_not_nan)


@partial(tf.function, experimental_relax_shapes=True)
def generic_filtering_element(F, Q, H, R, y):
    def _res_nan():
        A = F
        b = tf.zeros(F.shape[0], dtype=F.dtype)
        C = Q
        eta = tf.zeros_like(y)
        J = tf.zeros((y.shape[0], y.shape[0]), dtype=y.dtype)
        return A, b, C, J, eta

    def _res_not_nan():
        S = H @ mm(Q, H, transpose_b=True) + R
        chol = tf.linalg.cholesky(S)

        Kt = tf.linalg.cholesky_solve(chol, H @ Q)
        A = F - mm(Kt, H, transpose_a=True) @ F
        b = mv(Kt, y, transpose_a=True)
        C = Q - mm(Kt, H, transpose_a=True) @ Q

        HF = H @ F
        eta = mv(HF,
                 tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y, 1)), 1),
                 transpose_a=True)

        J = mm(HF, tf.linalg.cholesky_solve(chol, HF), transpose_a=True)
        return A, b, C, J, eta

    return tf.cond(tf.math.is_nan(y), _res_nan, _res_not_nan)

@partial(tf.function, experimental_relax_shapes=True)
def make_associative_filtering_elements(m0, P0, Fs, Qs, Hs, Rs, observations):
    first_elems = first_filtering_element(m0, P0, Fs[0], Qs[0], Hs[0], Rs[0], observations[0])
    generic_elems = tf.vectorized_map(lambda z: generic_filtering_element(*z),
                                      (Fs, Qs, Hs, Rs, observations), fallback_to_while_loop=False)
    return tuple(tf.concat([tf.expand_dims(first_e, 0), gen_es], 0)
                 for first_e, gen_es in zip(first_elems, generic_elems))


@partial(tf.function, experimental_relax_shapes=True)
def filtering_operator(elems):
    elem1, elem2 = elems

    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = A1.shape[0]
    I = tf.eye(dim, dtype=A1.dtype)

    temp = tf.linalg.solve(I + C1 @ J2, tf.transpose(A2), adjoint=True)
    A = mm(temp, A1, transpose_a=True)
    b = mv(temp, b1 + mv(C1, eta2), transpose_a=True) + b2
    C = mm(temp, mm(C1, A2, transpose_b=True), transpose_a=True) + C2

    temp = tf.linalg.solve(I + J2 @ C1, A1, adjoint=True)
    eta = mv(temp, eta2 - mv(J2, b1), transpose_a=True) + eta1
    J = mm(temp, J2 @ A1, transpose_a=True) + J1

    return A, b, C, J, eta


@partial(tf.function, experimental_relax_shapes=True)
def pkf(m0, P0, Fs, Qs, Hs, Rs, observations, return_loglikelihood=False, max_parallel=10000):
    initial_elements = make_associative_filtering_elements(m0, P0, Fs, Qs, Hs, Rs, observations)

    def vectorized_operator(a, b):
        return tf.vectorized_map(filtering_operator, (a, b), fallback_to_while_loop=False)

    final_elements = scan_associative(vectorized_operator,
                                      initial_elements,
                                      max_num_levels=math.ceil(math.log2(max_parallel)))
    if return_loglikelihood:
        predicted_means = tf.concat([[m0], final_elements[1][:-1]], axis=0)
        predicted_covs = mm(Fs, mm(tf.concat([[P0], final_elements[2][:-1]], axis=0), Fs, transpose_b=True)) + Qs
        obs_means = mm(Hs, predicted_means)
        obs_covs = mm(Hs, mm(predicted_covs, Hs, transpose_b=True))
        dists = MultivariateNormalFullCovariance(obs_means, obs_covs)
        logprobs = dists.log_prob(observations)
        return tf.reduce_sum(logprobs), final_elements[1], final_elements[2]
    return final_elements[1], final_elements[2]


@partial(tf.function, experimental_relax_shapes=True)
def last_smoothing_element(m, P):
    return tf.zeros_like(P), m, P


@partial(tf.function, experimental_relax_shapes=True)
def generic_smoothing_element(F, Q, m, P):
    Pp = F @ mm(P, F, transpose_b=True) + Q

    chol = tf.linalg.cholesky(Pp)
    E = tf.transpose(tf.linalg.cholesky_solve(chol, F @ P))
    g = m - mv(E @ F, m)
    L = P - E @ mm(Pp, E, transpose_b=True)
    return E, g, L


@partial(tf.function, experimental_relax_shapes=True)
def make_associative_smoothing_elements(Fs, Qs, filtering_means, filtering_covariances):
    last_elems = last_smoothing_element(filtering_means[-1], filtering_covariances[-1])
    generic_elems = tf.vectorized_map(lambda z: generic_smoothing_element(*z),
                                      (Fs[:-1], Qs[:-1], filtering_means[:-1], filtering_covariances[:-1]),
                                      fallback_to_while_loop=False)
    return tuple(tf.concat([gen_es, tf.expand_dims(last_e, 0)], axis=0)
                 for gen_es, last_e in zip(generic_elems, last_elems))


@partial(tf.function, experimental_relax_shapes=True)
def smoothing_operator(elems):
    elem1, elem2 = elems
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E2 @ E1
    g = mv(E2, g1) + g2
    L = E2 @ mm(L1, E2, transpose_b=True) + L2

    return E, g, L


@partial(tf.function, experimental_relax_shapes=True)
def pks(Fs, Qs, ms, Ps, max_parallel=10000):
    initial_elements = make_associative_smoothing_elements(Fs, Qs, ms, Ps)
    reversed_elements = tuple(tf.reverse(elem, axis=[0]) for elem in initial_elements)

    def vectorized_operator(a, b):
        return tf.vectorized_map(smoothing_operator, (a, b), fallback_to_while_loop=False)

    final_elements = scan_associative(vectorized_operator,
                                      reversed_elements,
                                      max_num_levels=math.ceil(math.log2(max_parallel)))
    return tf.reverse(final_elements[1], axis=[0]), tf.reverse(final_elements[2], axis=[0])


@partial(tf.function, experimental_relax_shapes=True)
def pkfs(model, observations, max_parallel=10000):
    return pks(model, *pkf(model, observations, max_parallel), max_parallel)
