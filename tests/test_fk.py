import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from functools import partial
from gfk.feynman_kac import compute_ess, smc_feynman_kac
from gfk.resampling import stratified
from gfk.filters import kf

jax.config.update('jax_enable_x64', True)


def test_ess():
    n = 10
    ws = jnp.ones(n) / n
    npt.assert_allclose(compute_ess(jnp.log(ws)), n * 1., rtol=1e-10)

    ws = jnp.zeros(n)
    ws = ws.at[1].set(1.)
    npt.assert_allclose(compute_ess(jnp.log(ws)), 1., rtol=1e-10)


def test_fk_gaussian():
    """Test FK-SMC vs a Kalman filter.
    """
    np.random.seed(666)
    key = jax.random.PRNGKey(666)
    key_sim, key_smc = jax.random.split(key)
    nparticles = 10000
    nsteps = 100

    state_trans = jnp.diag(jnp.array([0.5, 0.1, 0.2]))
    state_cov = jnp.eye(3)
    obs_trans = jnp.asarray(np.random.randn(2, 3))
    obs_cov = jnp.eye(2)
    mean0 = jnp.zeros(3)
    cov0 = jnp.eye(3)
    ys = jnp.asarray(np.random.randn(nsteps + 1, 2))

    mfs, vfs = kf(state_trans, state_cov, obs_trans, obs_cov, mean0, cov0, ys)

    def m0(key_):
        return mean0 + jax.random.normal(key_, (nparticles, 3)) @ jnp.linalg.cholesky(cov0)

    @partial(jax.vmap, in_axes=[0])
    def log_g0(xs):
        return jnp.sum(jax.scipy.stats.norm.logpdf(ys[0], obs_trans @ xs, 1.))

    def m(key_, xs, _):
        return (jnp.einsum('ij,...j->...i', state_trans, xs)
                + jax.random.normal(key_, (nparticles, 3)) @ jnp.linalg.cholesky(state_cov))

    @partial(jax.vmap, in_axes=[0, None, None])
    def log_g(xs, xs_prev, pytree):
        return jnp.sum(jax.scipy.stats.norm.logpdf(pytree[0], obs_trans @ xs, 1.))

    sampless, log_wss, esss = smc_feynman_kac(key_smc, m0, log_g0, m, log_g, (ys[1:],), nparticles, nsteps,
                                              resampling=stratified, resampling_threshold=0.9, return_path=True)
    # N, s, 2   N, s
    smc_m = jnp.einsum('ns...,ns->n...', sampless, jnp.exp(log_wss))
    smc_cov = jnp.einsum('nsi,nsj,ns->nij',
                         sampless - smc_m[:, None, :],
                         sampless - smc_m[:, None, :],
                         jnp.exp(log_wss))
    npt.assert_allclose(smc_m, mfs, atol=5e-2)
    npt.assert_allclose(smc_cov, vfs, atol=1e-1)
