import jax
import jax.numpy as jnp
import numpy.testing as npt
from gfk.likelihoods import pushfwd_normal, pushfwd_normal_batch

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
