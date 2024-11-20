import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gfk.synthetic_targets import make_gaussian_mixture, gm_lin_posterior
from gfk.tools import logpdf_gm, sampling_gm
from gfk.feynman_kac import make_fk_wu_normal
from gfk.resampling import stratified

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(1)

# Define the forward process
a, b = -0.5, 1.


def fwd_drift(x, t):
    return a * x


# Times
t0, T = 0., 1.
nsteps = 64
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the data
dim = 2
ws = jnp.array([0.1, 0.5, 0.2, 0.2])
ms = jnp.array([[2., 2.],
                [2., -2.],
                [-2., -2.],
                [-2., 2.]])
c1_ = jnp.array([[1., 0.2], [0.2, 0.5]])
c2_ = jnp.array([[1., -0.2], [-0.2, 0.5]])
c3_ = jnp.array([[0.5, 0.2], [0.2, 1.]])
c4_ = jnp.array([[2., 0.], [0., 1.]])
covs = jnp.concatenate([c1_[jnp.newaxis], c2_[jnp.newaxis], c3_[jnp.newaxis], c4_[jnp.newaxis]]) * 0.1
eigvals, eigvecs = jnp.linalg.eigh(covs)
wTs, mTs, eigvalTs, score, rev_drift, rev_dispersion = make_gaussian_mixture(ws, ms, eigvals, eigvecs, a, b, t0, T)

# Define the observation operator and the observation covariance
key, subkey = jax.random.split(key)
obs_op = jax.random.uniform(subkey, (1, dim))
obs_cov = jnp.eye(1)

y = jnp.ones(1)
posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    keys_ = jax.random.split(key_, num=nparticles)
    return jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, wTs, mTs, eigvalTs, eigvecs)


# Do conditional sampling
nparticles = 1024

# The
langevin_step_size = dt
smc_sampler = make_fk_wu_normal(obs_op, obs_cov,
                                rev_drift, rev_dispersion,
                                fwd_drift, score,
                                ts, y, langevin_step_size,
                                mode='guided',
                                proposal='langevin')

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
usT, log_wsT, esss = smc_sampler(subkey, m0, nparticles, stratified, 0.3, False)

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=256)
post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, posterior_ws, posterior_ms,
                                                                          eigvalTs, eigvecs)

grid = jnp.linspace(-8, 8, 1000)
meshgrid = jnp.meshgrid(grid, grid)
cartesian = jnp.dstack(meshgrid)
logpdf = lambda x_: logpdf_gm(x_, posterior_ws, posterior_ms, posterior_covs)
pdfs = jax.vmap(jax.vmap(logpdf))(cartesian)
plt.contour(*meshgrid, pdfs, levels=50)

plt.scatter(usT[:, 0], usT[:, 1], s=1)
plt.scatter(post_samples[:, 0], post_samples[:, 1], facecolors='none', edgecolors='tab:red', s=20)

plt.show()
