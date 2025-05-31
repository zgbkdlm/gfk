"""
Test the synthetic target distributions used in the experiments.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import sampling_gm, euler_maruyama
from functools import partial

jax.config.update('jax_enable_x64', True)


def test_gm():
    key = jax.random.PRNGKey(666)
    ncs, d = 3, 2

    ws = jnp.arange(1, ncs + 1)
    ws = ws / ws.sum()
    ms = jax.random.normal(key, (ncs, d)) * 1.5
    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (ncs, d))
    covs = jnp.einsum('ni,nj->nij', rnds, rnds) + jnp.eye(d)[None, :, :]
    eigvals, eigvecs = jnp.linalg.eigh(covs)

    def logpdf(x_, ms_, covs_, ws_):
        return jax.scipy.special.logsumexp(
            jax.vmap(jax.scipy.stats.multivariate_normal.logpdf, in_axes=[None, 0, 0])(x_, ms_, covs_), b=ws_)

    wTs, mTs, dTs, score, rev_drift, rev_dispersion = make_gm_bridge(ws, ms, eigvals, eigvecs, a=-0.5, b=1.,
                                                                     t0=0., T=1.)

    x = jnp.ones(d)
    npt.assert_allclose(score(x, 0.), jax.grad(logpdf)(x, ms, covs, ws))
    npt.assert_allclose(score(x, 1.), jax.grad(logpdf)(x, mTs,
                                                       jnp.einsum('nik,nk,njk->nij', eigvecs, dTs, eigvecs), wTs))

    nmcs = 10000
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=nmcs)
    xTs = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, wTs, mTs, dTs, eigvecs)

    ts = jnp.linspace(0., 1., 100)

    @partial(jax.vmap, in_axes=[0, 0])
    def reversal_sampler(key_, init):
        return euler_maruyama(key_, init, ts, rev_drift, rev_dispersion, integration_nsteps=10, return_path=False)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=nmcs)
    x0s = reversal_sampler(keys, xTs)

    npt.assert_allclose(jnp.mean(x0s, axis=0), jnp.sum(ws[:, None] * ms, axis=0), atol=5e-2)

    import matplotlib.pyplot as plt

    def pdf(x):
        return jnp.exp(logpdf(x, ms, covs, ws))

    grid = jnp.linspace(-8, 8, 1000)
    meshgrid = jnp.meshgrid(grid, grid)
    cartesian = jnp.dstack(meshgrid)
    pdfs = jax.vmap(jax.vmap(pdf))(cartesian)
    plt.contourf(*meshgrid, pdfs / pdfs.sum())
    plt.scatter(x0s[::10, 0], x0s[::10, 1], s=1, alpha=0.5)
    plt.scatter(ms[:, 0], ms[:, 1], c='r', s=10)
    plt.show()

    # Test posterior
    dy = 4
    key, subkey = jax.random.split(key)
    obs_op = jax.random.normal(subkey, (dy, d))
    obs_cov = jnp.eye(dy)
    y = jnp.zeros(dy)

    def log_energy(x_):
        return jax.scipy.stats.multivariate_normal.logpdf(y, obs_op @ x_, obs_cov) + logpdf(x_, ms, covs, ws)

    posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)

    def computed_posterior_logpdf(x_):
        return logpdf(x_, posterior_ms, posterior_covs, posterior_ws)

    x = jnp.ones(d)
    npt.assert_allclose(jax.grad(computed_posterior_logpdf)(x), jax.grad(log_energy)(x))
