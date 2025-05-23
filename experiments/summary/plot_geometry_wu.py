import numpy as np
import matplotlib.pyplot as plt
import math
import jax
import jax.numpy as jnp
from ot.sliced import sliced_wasserstein_distance
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import sampling_gm, logpdf_mvn_chol
from gfk.feynman_kac import make_fk_wu
from gfk.resampling import stratified
from gfk.experiments import generate_gm
from matplotlib.transforms import Bbox

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 20})

seed = 83
key_mc = np.load('rnd_keys.npy')[seed]
tweedie = True

# Define the forward process
a, b = -1., math.sqrt(2)

# Times
t0, T = 0., 2.
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the SMC conditional sampler
nparticles = 1024
nsamples = nparticles


# The sampler
@jax.jit
def sampler(key_, obs_op_, obs_cov_, y_, init, target):
    ws_, ms_, eigvals_, eigvecs_ = target
    *_, score, rev_drift, rev_dispersion = make_gm_bridge(ws_, ms_, eigvals_, eigvecs_, a, b, t0, T)
    chol = jnp.linalg.cholesky(obs_cov_)

    def cond_expec_tweedie(u, t):
        alp = jnp.exp(a * (T - t) / 2)
        return 1 / alp * (u - (1 - alp ** 2) * score(u, T - t))

    def cond_expec_euler(u, t):
        return u + rev_drift(u, t) * (T - t)

    smc = make_fk_wu(lambda y_, u: logpdf_mvn_chol(y_, obs_op_ @ u, chol),
                     rev_drift, rev_dispersion,
                     ts, y_, dt * 2,
                     cond_expec_tweedie if tweedie else cond_expec_euler,
                     mode='guided', proposal='direct', bypass_smc=False)
    samples_, log_ws_, esss_ = smc(key_, init, nparticles, stratified, 0.7, True)
    return samples_, log_ws_, esss_


# MC loop
key_model, key_algo = jax.random.split(key_mc)

# Generate the model and data
dx, dy = 10, 1
key_model, subkey = jax.random.split(key_model)
ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, 10, full_obs_cov=True)
eigvals, eigvecs = jnp.linalg.eigh(covs)
wTs, mTs, eigvalTs, *_ = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T)

# Define the observation operator and the observation covariance
y_likely = jnp.einsum('ij,kj,k->i', obs_op, ms, ws)
y = y_likely + 20
posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)

# Generate initial (reference) samples
key_model, subkey = jax.random.split(key_model)
keys_ = jax.random.split(subkey, num=nparticles)
init_particles = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, wTs, mTs, eigvalTs, eigvecs)

# Compute true samples
_, subkey = jax.random.split(key_model)
keys_ = jax.random.split(subkey, num=nparticles)
post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, posterior_ws, posterior_ms,
                                                                          posterior_eigvals, posterior_eigvecs)

# Run the SMC sampler
samples, log_ws, esss = sampler(key_algo, obs_op, obs_cov, y, init_particles, (ws, ms, eigvals, eigvecs))

# Compute errors
swd = sliced_wasserstein_distance(samples[-1], post_samples, a=jnp.exp(log_ws[-1]), n_projections=1000, p=1)
print(f'Sliced Wasserstein distance: {swd}')

nbins = 50
marginal_id = 0

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(projection='3d')
ax.set_box_aspect((2, 1, 1))
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axis._axinfo["grid"].update({'linestyle': '--', 'alpha': 0.3})

for i in range(0, nsteps + 1, nsteps // 10):
    hists, bin_edges = np.histogram(samples[i, :, marginal_id], weights=np.exp(log_ws[i]), bins=nbins, density=True)
    bin_widths = np.diff(bin_edges)
    ax.bar(bin_edges[:-1], hists, width=bin_widths, align='edge', zs=ts[i], zdir='x', color='black', alpha=0.5)

# Plot the true histogram
hists, bin_edges = np.histogram(post_samples[:, marginal_id], bins=nbins, density=True)
bin_widths = np.diff(bin_edges)
ax.bar(bin_edges[:-1], hists, width=bin_widths, align='edge', zs=ts[-1], zdir='x', color='tab:red', alpha=0.5)

ax.set_xlabel('Time $t$')
ax.set_ylabel('$x$')
ax.set_zlabel('Histogram')

ax.xaxis.labelpad=18
ax.yaxis.labelpad=10

plt.savefig('geometry_wu.pdf', transparent=True, bbox_inches=Bbox([[4.1, 1.7], [12.2, 7.8]]))
plt.show()
