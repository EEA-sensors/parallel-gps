import math
from functools import partial

import tensorflow as tf
from tensorflow_probability.python.distributions import MultivariateNormalTriL
from tensorflow_probability.python.math import scan_associative

__all__ = ["pkf", "pks", "pkfs"]

mv = tf.linalg.matvec
mm = tf.linalg.matmul


@tf.function
def first_filtering_element(m0, P0, F, Q, H, R, y):
    def _res_nan():
        A = tf.zeros_like(F)
        b = m0
        C = P0
        eta = tf.zeros_like(m0)
        J = tf.zeros_like(F)

        return A, b, C, J, eta

    def _res_not_nan():
        S1 = H @ mm(P0, H, transpose_b=True) + R
        S1_chol = tf.linalg.cholesky(S1)
        K1t = tf.linalg.cholesky_solve(S1_chol, H @ P0)

        A = tf.zeros_like(F)
        b = m0 + mv(K1t, y - mv(H, m0), transpose_a=True)
        C = P0 - mm(K1t, S1, transpose_a=True) @ K1t

        S = H @ mm(Q, H, transpose_b=True) + R
        chol = tf.linalg.cholesky(S)
        HF = H @ F
        eta = mv(HF,
                 tf.squeeze(tf.linalg.cholesky_solve(chol, tf.expand_dims(y, 1)), 1),
                 transpose_a=True)
        J = mm(HF, tf.linalg.cholesky_solve(chol, H @ F), transpose_a=True)

        return A, b, C, J, eta

    res = tf.cond(tf.math.is_nan(y), _res_nan, _res_not_nan)
    for elem in res:
        print(elem.shape)
    return res

@tf.function
def generic_filtering_element(F, Q, H, R, y):
    def _res_nan():
        A = F
        b = tf.zeros((tf.shape(F)[0],), dtype=F.dtype)
        C = Q
        eta = tf.zeros((tf.shape(F)[0],), dtype=F.dtype)
        J = tf.zeros_like(F)

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

    res = tf.cond(tf.math.is_nan(y), _res_nan, _res_not_nan)
    for elem in res:
        print(elem.shape)
    return res


@tf.function
def make_associative_filtering_elements(m0, P0, Fs, Qs, H, R, observations):
    shape = tf.shape(Fs)
    A0, b0, C0, J0, eta0 = first_filtering_element(m0, P0, Fs[0], Qs[0], H, R, observations[0])

    specialized_fun = lambda z: generic_filtering_element(z[0], z[1], H, R, z[2])
    with tf.name_scope("generic_filtering_element"):
        As, bs, Cs, Js, etas = tf.vectorized_map(specialized_fun,
                                                 (Fs, Qs, observations),
                                                 fallback_to_while_loop=False)

    As = tf.reshape(As, shape)
    bs = tf.reshape(bs, shape[:-1])
    Cs = tf.reshape(Cs, shape)
    Js = tf.reshape(Js, shape)
    etas = tf.reshape(etas, shape[:-1])

    # These reshapes are only a matter of making sure that the shape of the tensors is known at compilation time.
    # There could be a better way but I am not aware of it, and the operations above are virtually free.
    # It is not clear why exactly we are losing the shape...
    return tuple(tf.tensor_scatter_nd_update(gen_es, [[0]], tf.expand_dims(first_e, 0))
                 for first_e, gen_es in zip((A0, b0, C0, J0, eta0),
                                            (As, bs, Cs, Js, etas)))
    # return tuple(tf.concat([tf.expand_dims(first_e, 0), gen_es[1:]], 0)
    #              for first_e, gen_es in zip((A0, b0, C0, J0, eta0),
    #                                         (As, bs, Cs, Js, etas)))


@tf.function
def filtering_operator(elems):
    elem1, elem2 = elems
    A1, b1, C1, J1, eta1 = elem1
    A2, b2, C2, J2, eta2 = elem2
    dim = tf.shape(A1)[0]
    I = tf.eye(dim, dtype=A1.dtype)

    temp = tf.linalg.solve(I + C1 @ J2, tf.transpose(A2), adjoint=True)
    A = mm(temp, A1, transpose_a=True)
    b = mv(temp, b1 + mv(C1, eta2), transpose_a=True) + b2
    C = mm(temp, mm(C1, A2, transpose_b=True), transpose_a=True) + C2

    temp = tf.linalg.solve(I + J2 @ C1, A1, adjoint=True)
    eta = mv(temp, eta2 - mv(J2, b1), transpose_a=True) + eta1
    J = mm(temp, J2 @ A1, transpose_a=True) + J1
    return A, b, C, J, eta


@tf.function
def pkf(lgssm, observations, return_loglikelihood=False):
    with tf.name_scope("parallel_filter"):
        P0, Fs, Qs, H, R = lgssm
        dtype = P0.dtype
        m0 = tf.zeros(tf.shape(P0)[0], dtype=dtype)

        n_elements = observations.shape[0]
        max_num_levels = math.ceil(math.log2(n_elements)) - 1
        initial_elements = make_associative_filtering_elements(m0, P0, Fs, Qs, H, R, observations)

        def vectorized_operator(a, b):
            return tf.vectorized_map(filtering_operator, (a, b), fallback_to_while_loop=False)

        final_elements = scan_associative(vectorized_operator,
                                          initial_elements,
                                          max_num_levels=max_num_levels)

        if return_loglikelihood:
            with tf.name_scope("log-likelihood"):
                filtered_means = tf.concat([tf.expand_dims(m0, 0), final_elements[1][:-1]], axis=0)
                filtered_cov = tf.concat([tf.expand_dims(P0, 0), final_elements[2][:-1]], axis=0)
                predicted_means = mv(Fs, filtered_means)
                predicted_covs = mm(Fs, mm(filtered_cov, Fs, transpose_b=True)) + Qs
                obs_means = mv(H, predicted_means)
                obs_covs = mm(H, mm(predicted_covs, H, transpose_b=True)) + tf.expand_dims(R, 0)
                dists = MultivariateNormalTriL(obs_means, tf.linalg.cholesky(obs_covs))
                # TODO: some logic could be added here to avoid handling the covariance of non-nan models, but no impact for GPs
                logprobs = dists.log_prob(observations)
                logprobs_without_nans = tf.where(tf.math.is_nan(logprobs),
                                                 tf.zeros_like(logprobs),
                                                 logprobs)
            return final_elements[1], final_elements[2], tf.reduce_sum(logprobs_without_nans)
        return final_elements[1], final_elements[2]


@tf.function
def last_smoothing_element(m, P):
    return tf.zeros_like(P), m, P


@tf.function
def generic_smoothing_element(F, Q, m, P):
    Pp = F @ mm(P, F, transpose_b=True) + Q
    chol = tf.linalg.cholesky(Pp)
    E = tf.transpose(tf.linalg.cholesky_solve(chol, F @ P))
    g = m - mv(E @ F, m)
    L = P - E @ mm(Pp, E, transpose_b=True)
    return E, g, L


@tf.function
def make_associative_smoothing_elements(Fs, Qs, filtering_means, filtering_covariances):
    last_elems = last_smoothing_element(filtering_means[-1], filtering_covariances[-1])
    generic_elems = tf.vectorized_map(lambda z: generic_smoothing_element(*z),
                                      (Fs[1:], Qs[1:], filtering_means[:-1], filtering_covariances[:-1]),
                                      fallback_to_while_loop=False)
    return tuple(tf.concat([gen_es, tf.expand_dims(last_e, 0)], axis=0)
                 for gen_es, last_e in zip(generic_elems, last_elems))


@tf.function
def smoothing_operator(elems):
    elem1, elem2 = elems
    E1, g1, L1 = elem1
    E2, g2, L2 = elem2

    E = E2 @ E1
    g = mv(E2, g1) + g2
    L = E2 @ mm(L1, E2, transpose_b=True) + L2

    return E, g, L


@tf.function
def pks(lgssm, ms, Ps):
    n_elements = ms.shape[0]
    max_num_levels = math.ceil(math.log2(n_elements)) - 1
    _, Fs, Qs, *_ = lgssm
    initial_elements = make_associative_smoothing_elements(Fs, Qs, ms, Ps)
    reversed_elements = tuple(tf.reverse(elem, axis=[0]) for elem in initial_elements)

    def vectorized_operator(a, b):
        return tf.vectorized_map(smoothing_operator, (a, b), fallback_to_while_loop=False)

    final_elements = scan_associative(vectorized_operator,
                                      reversed_elements,
                                      max_num_levels=max_num_levels)
    return tf.reverse(final_elements[1], axis=[0]), tf.reverse(final_elements[2], axis=[0])


@tf.function
def pkfs(model, observations):
    fms, fPs = pkf(model, observations, False)
    return pks(model, fms, fPs)
