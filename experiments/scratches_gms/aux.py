import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from ott.tools.sliced import sliced_wasserstein
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import logpdf_gm, sampling_gm
from gfk.feynman_kac import make_fk_normal_likelihood
from gfk.resampling import stratified
from gfk.experiments import generate_gm

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(911)

# Define the forward prior process
a, b = -0.5, 1.


def fwd_drift(x, t):
    return a * x


# Times
t0, T = 0., 4
nsteps = 256
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the data
key, subkey = jax.random.split(key)
dx, dy = 10, 2
ncomponents = 5
ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, ncomponents)
eigvals, eigvecs = jnp.linalg.eigh(covs)
wTs, mTs, eigvalTs, score, rev_drift, rev_dispersion = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T)

# Define the observation operator and the observation covariance
y = jnp.ones(dy) * 5
posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    keys_ = jax.random.split(key_, num=nparticles)
    return jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, wTs, mTs, eigvalTs, eigvecs)


# Define the auxiliary process
aux_a, aux_b = -0.5, 1.


def aux_trans_op(i):
    return jnp.exp(aux_a * (ts[i + 1] - ts[i]))


def aux_semigroup(n, m):
    return jnp.exp(aux_a * (ts[n] - ts[m]))


def aux_trans_var(i):
    return aux_b ** 2 / (2 * aux_a) * (jnp.exp(2 * aux_a * (ts[i + 1] - ts[i])) - 1)


ys = jax.vmap(lambda n: aux_semigroup(n, 0) * y, in_axes=0)(jnp.arange(nsteps + 1))
vs = ys[::-1]

# Do conditional sampling
nparticles = 1024

# The sampler
smc_sampler, _ = make_fk_normal_likelihood(obs_op, obs_cov, rev_drift, rev_dispersion,
                                           aux_trans_op, aux_semigroup, aux_trans_var,
                                           ts, mode='guided')

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
usT, log_wsT, esss = smc_sampler(subkey, m0, vs, nparticles, stratified, 0.7, False)
plt.plot(esss)
plt.show()

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num=nparticles)
post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys, posterior_ws, posterior_ms,
                                                                          posterior_eigvals, posterior_eigvecs)

print(sliced_wasserstein(usT, post_samples, a=jnp.exp(log_wsT))[0])

key, subkey = jax.random.split(key)
inds = stratified(subkey, jnp.exp(log_wsT))
plt.scatter(usT[inds, 0], usT[inds, 1], s=1)
plt.scatter(post_samples[:, 0], post_samples[:, 1], facecolors='none', edgecolors='tab:red', s=20, alpha=.3)
plt.show()
