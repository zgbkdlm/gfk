import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ott.tools.sliced import sliced_wasserstein
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import logpdf_gm, sampling_gm, logpdf_mvn
from gfk.feynman_kac import make_fk_bridge
from gfk.resampling import stratified
from gfk.experiments import generate_gm

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(7)

# Define the forward prior process
a, b = -1, 1.

# Times
t0, T = 0., 5
nsteps = 500
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the data
key, subkey = jax.random.split(key)
dx, dy = 10, 1
ncomponents = 5
ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, ncomponents)
eigvals, eigvecs = jnp.linalg.eigh(covs)
wTs, mTs, eigvalTs, score, rev_drift, rev_dispersion = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T)

# Define the observation operator and the observation covariance
y = jnp.ones(dy) * 5
posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)

obs_eigvals, obs_eigvecs = jnp.linalg.eigh(obs_cov)


def logpdf_target_ll(y_, x):
    return logpdf_mvn(y_, obs_op @ x, obs_eigvals, obs_eigvecs)


# Define the reference likelihood
# Design a one that is as informative as possible
def logpdf_ref_ll(y_, x):
    return logpdf_mvn(y_, obs_op @ x, obs_eigvals * jnp.exp(a * T), obs_eigvecs)


# Define the interpolation process
def alpha(t):
    return jax.lax.cond(t == T,
                        lambda _: 1.,
                        lambda _: 1 - jnp.exp(-4. * t / T),
                        None)


# Generate an interpolation observations
y_ref = jnp.einsum('ij,kj,k->i', obs_op, mTs, wTs)
vs = jax.vmap(lambda t_: (1 - alpha(t_)) * y_ref + alpha(t_) * y)(ts)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    keys_ = jax.random.split(key_, num=nparticles)
    return jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, wTs, mTs, eigvalTs, eigvecs)


# Do conditional sampling
nparticles = 4096

# The sampler
langevin_step_size = dt * 2
smc_sampler = make_fk_bridge(logpdf_target_ll, logpdf_ref_ll, alpha,
                             rev_drift, rev_dispersion,
                             ts, mode='guided')

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
uss, log_wss, esss = smc_sampler(subkey, m0, vs, nparticles, stratified, 1., True)
plt.plot(esss)
plt.show()

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nparticles)
post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, posterior_ws, posterior_ms,
                                                                          posterior_eigvals, posterior_eigvecs)

print(sliced_wasserstein(uss[-1], post_samples, a=jnp.exp(log_wss[-1]))[0])

nbins = 50
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(0, nsteps + 1, nsteps // 16):
    hists, bin_edges = np.histogram(uss[i, :, 0], weights=jnp.exp(log_wss[i]), bins=50, density=True)
    bin_widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], hists, width=bin_widths, align='edge', zs=ts[i], zdir='x', color='black', alpha=0.5)

# Plot the true histogram
hists, bin_edges = np.histogram(post_samples[:, 0], bins=50, density=True)
bin_widths = np.diff(bin_edges)
ax.bar(bin_edges[:-1], hists, width=bin_widths, align='edge', zs=ts[-1], zdir='x', color='tab:red', alpha=0.5)

plt.tight_layout(pad=0.1)
plt.show()
