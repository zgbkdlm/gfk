import jax
import jax.numpy as jnp
import numpy.testing as npt
from gfk.likelihoods import pushfwd_normal, pushfwd_normal_batch, true_pushfwd_normal

jax.config.update("jax_enable_x64", True)


def test_gaussian():
    key = jax.random.PRNGKey(666)
    H = jax.random.normal(key, (5, 3))

    key, subkey = jax.random.split(key)
    r_ = jax.random.normal(subkey, (5,))
    R = jnp.outer(r_, r_) + jnp.eye(5)

    nsteps = 100
    ts = jnp.linspace(0., 1., nsteps + 1)

    def A(i):
        return jnp.exp(-0.5 * (ts[i + 1] - ts[i]))

    def S(n, m):
        return jnp.exp(-0.5 * (ts[n] - ts[m]))

    def C(i):
        return 0.1 * i

    def Sigma(i):
        return 1. - jnp.exp(-(ts[i + 1] - ts[i]))

    @jax.jit
    def pfwd(i):
        return pushfwd_normal(H, R, S, Sigma, C, i)

    Fs, Omegas = pushfwd_normal_batch(H, R, A, Sigma, C, nsteps)

    # Test if the two routines match
    for i in range(nsteps + 1):
        F, Omega = pfwd(i)
        npt.assert_allclose(F, Fs[i])
        npt.assert_allclose(Omega, Omegas[i])
        true_F = S(i, 0) * H
        true_Omega = S(i, 0) ** 2 * (H @ H.T * sum([C(i) for i in range(1, i + 1)]) + R) + (
                1. - jnp.exp(-(ts[i] - ts[0])))
        npt.assert_allclose(F, true_F)
        npt.assert_allclose(Omega, true_Omega)


def test_true_pushfwd_normal():
    key = jax.random.PRNGKey(666)
    m = jnp.array([1.1, 2.1])
    v = jnp.array([[1., 0.2],
                   [0.2, 1.5]])
    obs_op = jax.random.normal(key, (3, 2))
    obs_cov = jnp.array([[1., 0.1, -0.1],
                         [0.1, 2., 0.2],
                         [-0.1, 0.2, 1.7]])

    @jax.jit
    def ll(y_, x_, t):
        return true_pushfwd_normal(y_, x_, t, m, v, obs_op, obs_cov)

    # The likelihood function should coincide at 0
    key, subkey = jax.random.split(key)
    y = jax.random.normal(subkey, (3,))
    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (2,))

    npt.assert_allclose(ll(y, x, 0.), jax.scipy.stats.multivariate_normal.logpdf(y, obs_op @ x, obs_cov), rtol=1e-10)

    # The likelihood should become whittened as t -> inf
    npt.assert_allclose(ll(y, x, 20.), jax.scipy.stats.multivariate_normal.logpdf(y, jnp.zeros(3), jnp.eye(3)),
                        rtol=1e-4)
