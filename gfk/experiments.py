"""
This contains the common experiment settings.
"""
import jax
import jax.numpy as jnp
# import torch
# from functools import partial



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
    # obs_cov = (jnp.outer(obs_cov_rnds, obs_cov_rnds) + jnp.eye(dy)) * 1.
    obs_cov = jnp.eye(dy)
    return ws, ms, covs, obs_op, jnp.diag(jnp.diag(obs_cov)) if diag_obs_cov else obs_cov


# def filip(seed, dim_x, dim_y):
#     def ou_mixt(alpha_t, means, dim, weights):
#         cat = torch.distributions.Categorical(weights, validate_args=False)
#
#         ou_norm = torch.distributions.MultivariateNormal(
#             torch.vstack(tuple((alpha_t ** .5) * m for m in means)),
#             torch.eye(dim, device=means[0].device).repeat(len(means), 1, 1), validate_args=False)
#         return torch.distributions.MixtureSameFamily(cat, ou_norm, validate_args=False)
#
#     def build_extended_svd(A: torch.tensor):
#         U, d, V = torch.linalg.svd(A, full_matrices=True)
#         coordinate_mask = torch.ones_like(V[0])
#         coordinate_mask[len(d):] = 0
#         return U, d, coordinate_mask, V
#
#     def generate_measurement_equations(dim, dim_y, mixt, device):
#         A = torch.randn((dim_y, dim), device=device)
#
#         u, diag, coordinate_mask, v = build_extended_svd(A)
#         diag = torch.sort(torch.rand_like(diag), descending=True).values
#
#         A = u @ (torch.diag(diag) @ v[coordinate_mask == 1, :])
#         init_sample = mixt.sample()
#         std = (torch.rand((1,)))[0] * max(diag)
#         var_observations = std ** 2
#
#         init_obs = A @ init_sample
#         init_obs += torch.randn_like(init_obs) * std
#         return A.to(device), var_observations.to(device), init_obs.to(device)
#
#     def setup_of_gmm(seed, dim_x, dim_y, device="cpu"):
#         random_state = seed
#         device = device
#         torch.manual_seed(random_state)
#
#         # setup of the inverse problem
#         means = []
#         for i in range(-2, 3):
#             means += [torch.tensor([-8. * i, -8. * j] * (dim_x // 2), device=device) for j in range(-2, 3)]
#         weights = torch.randn(len(means), device=device) ** 2
#         weights = weights / weights.sum()
#         ou_mixt_fun = partial(ou_mixt,
#                               means=means,
#                               dim=dim_x,
#                               weights=weights)
#
#         mixt = ou_mixt_fun(1)
#
#         A, var_observations, init_obs = generate_measurement_equations(dim_x, dim_y, mixt, device)
#         return (jnp.asarray(weights), jnp.concatenate([jnp.asarray(m)[None, :] for m in means], axis=0), jnp.asarray(A),
#                 jnp.asarray(var_observations) * jnp.eye(1), jnp.asarray(init_obs))
#     return setup_of_gmm(seed, dim_x, dim_y)
