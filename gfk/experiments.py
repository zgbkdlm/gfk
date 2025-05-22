"""
This contains the common experiment settings.
"""
import jax
import jax.numpy as jnp
import itertools
import math


def generate_gm(key, dx, dy, ncomponents, full_obs_cov: bool = False, noiseless: bool = False):
    """Generate a GM (with observation) model.
    This is similar to the one used in the MCGDiff paper but is more challenging.
    But this has more randomness in the setting; may require more MC runs for convincing results.
    """
    key, subkey = jax.random.split(key)
    ws = jax.random.normal(subkey, shape=(ncomponents,)) ** 2
    ws = ws / jnp.sum(ws)

    key, subkey = jax.random.split(key)
    ms = jax.random.uniform(subkey, minval=-8, maxval=8., shape=(ncomponents, dx))

    key, subkey = jax.random.split(key)
    cov_rnds = jax.random.uniform(subkey, shape=(ncomponents, dx))
    covs = (jnp.einsum('...i,...j->...ij', cov_rnds, cov_rnds) + jnp.eye(dx)[None, :, :])

    key, subkey = jax.random.split(key)
    obs_op = jax.random.normal(subkey, shape=(dy, dx))
    u, s, vh = jnp.linalg.svd(obs_op, full_matrices=False)

    key, subkey = jax.random.split(key)
    s = jnp.sort(jax.random.uniform(subkey, (dy,)), descending=True) + 1e-3
    # s = jnp.ones(dy)
    obs_op = u @ jnp.diag(s) @ vh

    key, subkey = jax.random.split(key)
    if full_obs_cov:
        obs_cov_rnds = jax.random.uniform(subkey, shape=(dy,))
        obs_cov = jnp.outer(obs_cov_rnds, obs_cov_rnds) + jnp.eye(dy) * max(s) ** 2
    else:
        obs_cov = jnp.eye(dy) * jax.random.uniform(subkey) * jnp.max(s) ** 2

    if noiseless:
        obs_cov = jnp.eye(dy) * 1e-8
    return ws, ms, covs, obs_op, obs_cov


# def generate_gm_mcgdiff(key, dx, dy, ncomponents=25):
#     """Generate a GM (with observation) model.
#     """
#     if dx % 2 != 0:
#         raise ValueError("dx must be even.")
#
#     key, subkey = jax.random.split(key)
#     ws = jax.random.normal(subkey, shape=(ncomponents,)) ** 2
#     ws = ws / jnp.sum(ws)
#
#     a = (-2, -1, 0, 1, 2)
#     ms_ = [jnp.tile(jnp.array([8. * i, 8. * j]), (1, dx // 2)) for (i, j) in itertools.product(a, a)]
#     ms = jnp.concatenate(ms_, axis=0)
#     covs = jnp.tile(jnp.eye(dx), (ncomponents, 1, 1))
#
#     key, subkey = jax.random.split(key)
#     obs_op = jax.random.normal(subkey, shape=(dy, dx))
#     u, s, vh = jnp.linalg.svd(obs_op, full_matrices=False)
#     key, subkey = jax.random.split(key)
#     s = jnp.sort(jax.random.uniform(subkey, (dy,)), descending=True) + 1
#     obs_op = u @ jnp.diag(s) @ vh
#
#     key, subkey = jax.random.split(key)
#     obs_cov = jnp.eye(dy) * jax.random.uniform(subkey) * jnp.max(s)
#     return ws, ms, covs, obs_op, obs_cov
