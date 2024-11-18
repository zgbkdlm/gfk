import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
from gfk.synthetic_targets import make_gsb
from gfk.tools import bures
from gfk.feynman_kac import make_fk_normal_likelihood
from gfk.resampling import stratified
from functools import partial

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(6)

# Define the data
dim = 10
m_ref, cov_ref, mT, covT, drift, dispersion, log_likelihood, obs_op, obs_cov, posterior_m_cov = make_gsb(key, d=dim)
chol_ref = jnp.linalg.cholesky(cov_ref)
y = jnp.zeros(dim)

# Times
T = 1.
nsteps = 128
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)


def ref_sampler(key_, n: int = 1):
    """The reference distribution is a standard Normal.
    """
    return m_ref + jnp.einsum('ij,nj->ni', chol_ref, jax.random.normal(key_, shape=(n, dim)))


def ref_log_likelihood(y, x):
    pass  # TODO


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    return ref_sampler(key_, n=nparticles)


# Define the auxiliary process
def aux_trans_op(i):
    return jnp.exp(-0.5 * (ts[i + 1] - ts[i]))


def aux_semigroup(n, m):
    return jnp.exp(-0.5 * (ts[n] - ts[m]))


def aux_trans_var(i):
    return 1. - jnp.exp(-(ts[i + 1] - ts[i]))


ys = jax.vmap(lambda n: aux_semigroup(n, 0) * y, in_axes=0)(jnp.arange(nsteps + 1))
vs = ys[::-1]

# Do conditional sampling
nparticles = 10000

# The
smc_sampler = make_fk_normal_likelihood(obs_op, obs_cov, drift, dispersion, aux_trans_op, aux_semigroup, aux_trans_var,
                                        ts)

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
usT, log_wsT, esss = smc_sampler(subkey, m0, vs, nparticles, stratified, 0.3, False)

key, subkey = jax.random.split(key)

post_m, post_cov = posterior_m_cov(y)
approx_m = jnp.einsum('si,s->i', usT, jnp.exp(log_wsT))
approx_cov = jnp.einsum('si,sj,s->ij', usT - approx_m, usT - approx_m, jnp.exp(log_wsT))

print(bures(post_m, post_cov, approx_m, approx_cov))

plt.plot(esss)
plt.show()

plt.plot(post_m)
plt.plot(approx_m)
plt.show()
