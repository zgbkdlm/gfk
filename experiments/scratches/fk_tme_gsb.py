import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import tme.base_jax as tme
from gfk.synthetic_targets import make_gsb
from gfk.tools import bures
from gfk.feynman_kac import smc_feynman_kac
from gfk.resampling import stratified
from functools import partial

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(6)

# Define the data
dim = 10
m_ref, cov_ref, mT, covT, drift, dispersion, log_likelihood, posterior_m_cov = make_gsb(key, d=dim)
chol_ref = jnp.linalg.cholesky(cov_ref)
y = jnp.zeros(dim)

# Times
T = 1.
nsteps = 128
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

nblocks = 1
block_dt = dt * (nsteps / nblocks)
block_ts = jnp.linspace(0, T, nblocks + 1)


def ref_sampler(key_, n: int = 1):
    """The reference distribution is a standard Normal.
    """
    return m_ref + jnp.einsum('ij,nj->ni', chol_ref, jax.random.normal(key_, shape=(n, dim)))


def dispersion_d(u, t):
    return jnp.eye(dim)


def transition_sampler(key_, us, t_k):
    """The Euler--Maruyama transition of the reversal
    """
    cond_m, cond_scale = us + jax.vmap(drift, in_axes=[0, None])(us, t_k) * dt, math.sqrt(dt) * dispersion(us, t_k)
    return cond_m + cond_scale * jax.random.normal(key_, shape=us.shape)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    return ref_sampler(key_, n=nparticles)


def log_g0(us):
    return log_lk(us, ts[0])


def step_fn(t):
    cond_list = [t == 0.] + [(t > lb) & (t <= ub) for (lb, ub) in zip(block_ts[:-1], block_ts[1:])]
    func_list = [lambda _: block_ts[1]] + [lambda _, parg=block_t: parg for block_t in block_ts[1:]]
    return jnp.piecewise(t, cond_list, func_list)


@partial(jax.vmap, in_axes=[0, None])
def log_lk(us, t_k):
    def phi(x, _):
        return log_likelihood(y, x)

    block_t = step_fn(t_k)
    return tme.expectation(phi, us, t_k, block_t - t_k, drift, dispersion_d, order=0)


def m(key_, us, tree_param):
    t_km1 = tree_param[0]
    return transition_sampler(key_, us, t_km1)


def log_g(us_k, us_km1, tree_param):
    t_km1, t_k = tree_param
    return log_lk(us_k, t_k) - log_lk(us_km1, t_km1)


# Do conditional sampling
nparticles = 1000

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
usT, log_wsT, esss = smc_feynman_kac(subkey, m0, log_g0, m, log_g, (ts[:-1], ts[1:]), nparticles, nsteps,
                                     stratified,
                                     0.5,
                                     False)

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
