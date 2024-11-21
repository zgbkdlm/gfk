"""
This contains the common experiment settings.
"""
import jax
import jax.numpy as jnp


def generate_gm(key, dx, dy, ncomponents):
    """Generate a GM (with observation) model.
    """
    key_ws, key_ms, key_covs, key_obs_op, key_obs_cov = jax.random.split(key, num=5)
    ws = jax.random.beta(key_ws, a=3., b=3., shape=(ncomponents,))
    ws = ws / jnp.sum(ws)

    ms = jax.random.uniform(key_ms, minval=-5., maxval=5., shape=(ncomponents, dx))

    cov_rnds = jax.random.normal(key_covs, shape=(ncomponents, dx))
    covs = (jnp.einsum('...i,...j->...ij', cov_rnds, cov_rnds) + jnp.eye(dx)[None, :, :])

    obs_op = jax.random.uniform(key_obs_op, shape=(dy, dx))
    obs_cov_rnds = jax.random.normal(key_covs, shape=(dy, ))
    obs_cov = (jnp.outer(obs_cov_rnds, obs_cov_rnds) + jnp.eye(dy)) * 1.
    return ws, ms, covs, obs_op, obs_cov
