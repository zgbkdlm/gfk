"""
This contains the common experiment settings.
"""
import jax
import jax.numpy as jnp


def generate_gm(key, dx, dy, ncomponents, diag_obs_cov: bool = False):
    """Generate a GM (with observation) model.
    """
    key_ws, key_ms, key_covs, key_obs_op, key_obs_cov = jax.random.split(key, num=5)
    ws = jax.random.beta(key_ws, a=3., b=3., shape=(ncomponents,))
    ws = ws / jnp.sum(ws)

    ms = jax.random.uniform(key_ms, minval=5., maxval=15., shape=(ncomponents, dx))

    cov_rnds = jax.random.normal(key_covs, shape=(ncomponents, dx))
    covs = (jnp.einsum('...i,...j->...ij', cov_rnds, cov_rnds) + jnp.eye(dx)[None, :, :])

    obs_op = jax.random.normal(key_obs_op, shape=(dy, dx))
    u, s, vh = jnp.linalg.svd(obs_op, full_matrices=False)
    _, subkey = jax.random.split(key_obs_op)
    s = jax.random.normal(subkey, (dy, ))
    obs_op = u @ jnp.diag(s) @ vh
    obs_cov_rnds = jax.random.normal(key_covs, shape=(dy, ))
    obs_cov = (jnp.outer(obs_cov_rnds, obs_cov_rnds) + jnp.eye(dy)) * 1.
    return ws, ms, covs, obs_op, jnp.diag(jnp.diag(obs_cov)) if diag_obs_cov else obs_cov
