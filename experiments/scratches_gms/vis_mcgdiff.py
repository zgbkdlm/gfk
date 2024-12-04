import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ott.tools.sliced import sliced_wasserstein
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import logpdf_gm, sampling_gm
from gfk.feynman_kac import make_mcgdiff
from gfk.resampling import stratified
from gfk.experiments import generate_gm

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(999)

# Define the forward process
a, b = -1., 1.

# Times
t0, T = 0., 5
nsteps = 500
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the data
key, subkey = jax.random.split(key)
dx, dy = 10, 1
ncomponents = 5
ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, ncomponents, full_obs_cov=True)
eigvals, eigvecs = jnp.linalg.eigh(covs)
wTs, mTs, eigvalTs, score, rev_drift, rev_dispersion = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T)

# Define the observation operator and the observation covariance
y_likely = jnp.einsum('ij,kj,k->i', obs_op, ms, ws)
y = y_likely + 0
posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    keys_ = jax.random.split(key_, num=nparticles)
    return jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, wTs, mTs, eigvalTs, eigvecs)


def alpha(t):
    return jnp.exp(a * (T - t))


# Do conditional sampling
nparticles = 1024
kappa = 1e-3

# The sampler
smc_sampler = make_mcgdiff(obs_op, obs_cov, rev_drift, rev_dispersion, alpha, y, ts, kappa, mode='guided')

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
ts_smc, ts_is, uss, log_wss, esss = smc_sampler(subkey, m0, nparticles, stratified, 0.7, True, full_final=False)

plt.plot(ts, esss)
plt.vlines(ts_smc[-1], esss.min(), esss.max(), colors='black', linestyles='--')
plt.show()

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nparticles)
post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, posterior_ws, posterior_ms,
                                                                          posterior_eigvals, posterior_eigvecs)

print(sliced_wasserstein(uss[-1], post_samples, a=jnp.exp(log_wss[-1]))[0])

plt.scatter(post_samples[:, 0], post_samples[:, 1], color='tab:red', alpha=0.5, s=1)
plt.show()

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
