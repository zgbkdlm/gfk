"""
The test problem in the pedagogical example.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
from gfk.tools import logpdf_mvn
from gfk.typings import JArray, JKey, FloatScalar
from typing import Tuple


def make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T):
    def drift(x, t):
        return a * x

    def dispersion(t):
        return b

    def logpdf_kth(x, m, u, d, t):
        dt = t - t0
        pushfwd_m = jnp.exp(a * dt) * m
        pushfwd_d = jnp.exp(2 * a * dt) * d + (b ** 2 / (2 * a) * (jnp.exp(2 * a * dt) - 1))
        return logpdf_mvn(x, pushfwd_m, pushfwd_d, u)

    def logpdf(x, t):
        return jax.scipy.special.logsumexp(
            jax.vmap(logpdf_kth, in_axes=[None, 0, 0, 0, None])(x, ms, eigvecs, eigvals, t), b=ws)

    def score(x, t):
        return jax.grad(logpdf)(x, t)

    def rev_drift(u, t):
        return -drift(u, T - t) + dispersion(T - t) ** 2 * score(u, T - t)

    def rev_dispersion(t):
        return dispersion(T - t)

    wTs = ws
    mTs = jnp.exp(a * (T - t0)) * ms
    eigvalTs = jnp.exp(2 * a * (T - t0)) * eigvals + (b ** 2 / (2 * a) * (jnp.exp(2 * a * (T - t0)) - 1))

    return wTs, mTs, eigvalTs, score, rev_drift, rev_dispersion


def gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs):
    """Compute the posterior distribution of a Gaussian mixture with linear Gaussian likelihood.
    """

    def single_posterior(w, m, cov):
        g = obs_op @ cov @ obs_op.T + obs_cov
        chol = jax.scipy.linalg.cho_factor(g)
        return (jnp.log(w) + jax.scipy.stats.multivariate_normal.logpdf(y, obs_op @ m, g),
                m + cov @ obs_op.T @ jax.scipy.linalg.cho_solve(chol, y - obs_op @ m),
                cov - cov @ obs_op.T @ jax.scipy.linalg.cho_solve(chol, obs_op @ cov))

    log_ws_, posterior_ms, posterior_covs = jax.vmap(single_posterior, in_axes=[0, 0, 0])(ws, ms, covs)
    return jnp.exp(log_ws_ - jax.scipy.special.logsumexp(log_ws_)), posterior_ms, posterior_covs


class BiochemicalO2Demand:
    pass
