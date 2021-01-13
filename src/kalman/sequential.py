from functools import partial

import tensorflow as tf

__all__ = ["kf", "ks", "kfs"]

from tensorflow_probability.python.distributions import MultivariateNormalTriL

mv = tf.linalg.matvec
mm = tf.linalg.matmul


@partial(tf.function, experimental_relax_shapes=True)
def kf(m0, P0, Fs, Qs, Hs, Rs, observations, return_loglikelihood=False):
    def body(carry, inp):
        ell, m, P = carry
        y, F, Q, H, R = inp
        m = mv(F, m)
        P = F @ mm(P, F, transpose_b=True) + Q

        S = H @ mm(P, H, transpose_b=True) + R
        yp = mv(H, m)
        chol = tf.linalg.cholesky(S)
        predicted_dist = MultivariateNormalTriL(yp, chol)
        ell_t = predicted_dist.log_prob(y)
        Kt = tf.linalg.cholesky_solve(chol, H @ P)

        m = m + mv(Kt, y - yp, transpose_a=True)
        P = P - mm(Kt, S, transpose_a=True) @ Kt
        return ell + ell_t, m, P

    ells, fms, fPs = tf.scan(body,
                             (observations, Fs, Qs, Hs, Rs),
                             (0., m0, P0))
    if return_loglikelihood:
        return ells[-1], fms, fPs
    return fms, fPs


@partial(tf.function, experimental_relax_shapes=True)
def ks(Fs, Qs, ms, Ps):
    def body(carry, inp):
        m, P, F, Q = inp
        sm, sP = carry

        pm = mv(F, m)
        pP = F @ mm(P, F, transpose_b=True) + Q

        chol = tf.linalg.cholesky(pP)
        Ct = tf.linalg.cholesky_solve(chol, F @ P)

        sm = m + mv(Ct, (sm - pm), transpose_a=True)
        sP = P + mm(Ct, sP - pP, transpose_a=True) @ Ct
        return sm, sP

    (sms, sPs) = tf.scan(body, (Fs[:-1], Qs[:-1], ms[:-1], Ps[:-1]), (ms[-1], Ps[-1]), reverse=True)
    sms = tf.concat([sms, tf.expand_dims(ms[-1], 0)], 0)
    sPs = tf.concat([sPs, tf.expand_dims(Ps[-1], 0)], 0)
    return sms, sPs


@partial(tf.function, experimental_relax_shapes=True)
def kfs(model, observations):
    return ks(model, *kf(model, observations))
