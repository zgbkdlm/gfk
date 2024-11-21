import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gfk.tools import logpdf_gm, sampling_gm, logpdf_mvn
from gfk.experiments import generate_gm

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(911)

# Define the forward prior process
a, b = -0.5, 1.

# Times
t0, T = 0., 1.
nsteps = 64
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the data
key, subkey = jax.random.split(key)
dx, dy = 2, 2
ncomponents = 10
ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, ncomponents)
eigvals, eigvecs = jnp.linalg.eigh(covs)

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=10000)

samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, ws, ms, eigvals, eigvecs)

key, subkey = jax.random.split(key)
ys = (jnp.einsum('ij,...j->...i', obs_op, samples)
      + jax.random.normal(subkey, (10000, dy)) @ jnp.linalg.cholesky(obs_cov).T)

plt.scatter(samples[:, 0], samples[:, 1], s=1)
plt.scatter(ys[:, 0], ys[:, 1], s=1)
plt.show()