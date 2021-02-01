import tensorflow as tf
import gpflow.config as config


def solve_lyap_vec(F: tf.Tensor,
                   L: tf.Tensor,
                   q: tf.Tensor) -> tf.Tensor:
    """Vectorized Lyapunov equation solver

    F P + P F' + L q L' = 0

    Parameters
    ----------
    F : tf.Tensor
        ...
    L : tf.Tensor
        ...
    q : tf.Tensor
        ...

    Returns
    -------
    Pinf : tf.Tensor
        Steady state covariance

    """
    dtype = config.default_float()

    dim = tf.shape(F)[0]

    op1 = tf.linalg.LinearOperatorFullMatrix(F)
    op2 = tf.linalg.LinearOperatorIdentity(dim, dtype=dtype)

    F1 = tf.linalg.LinearOperatorKronecker([op2, op1]).to_dense()
    F2 = tf.linalg.LinearOperatorKronecker([op1, op2]).to_dense()

    F = F1 + F2

    Q = L @ tf.transpose(L) * q

    Pinf = tf.reshape(tf.linalg.solve(F, tf.reshape(Q, (-1, 1))), (dim, dim))
    Pinf = -0.5 * (Pinf + tf.transpose(Pinf))
    return Pinf