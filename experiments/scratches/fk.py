import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
from gfk.synthetic_targets import Crescent
from gfk.nns import make_nn, CrescentMLP
from gfk.feynman_kac import smc_feynman_kac
from gfk.resampling import stratified

jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_disable_jit", True)
key = jax.random.PRNGKey(666)

# Define the data
crescent = Crescent(c=1., xi=0.5)
y_ref = crescent.emission(crescent.mean)


# The likelihood pi(y | x)
def logpdf_likelihood(y_, x):
    return crescent.logpdf_y_cond_x(y_, x)


def ref_log_likelihood(y_, x):
    return jax.scipy.stats.norm.logpdf(y_, jnp.sum(x), 1.)


def proxy_log_likelihood(x, t):
    alpha = t / T
    return crescent.logpdf_y_cond_x((1 - alpha) * y_ref + alpha * y, x)


# Load the DSB model for pi_X
# Define the parametric neural network
nn_dt = 1. / 200
key, subkey = jax.random.split(key)
my_nn = CrescentMLP(dt=nn_dt, dim_out=2)
_, _, nn_drift = make_nn(subkey, neural_network=my_nn, shape_x=(2,), shape_t=())
param_bwd = np.load('../checkpoints/dsb-x-15.npz')['param_bwd']

# Times
T = 1.
nsteps = 128
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)


def ref_sampler(key_, n: int = 1):
    """The reference distribution is a standard Normal.
    """
    return jax.random.normal(key_, shape=(n, 2))


def rev_drift(u, t):
    """The reversal part of the prior diffusion.
    """
    return nn_drift(u, T - t, param_bwd)


def rev_dispersion(_):
    return 1.


def rev_transition_sampler(key_, us, t_k):
    """The Euler--Maruyama transition of the reversal
    """
    cond_m, cond_scale = us + rev_drift(us, t_k) * dt, math.sqrt(dt) * rev_dispersion(t_k)
    return cond_m + cond_scale * jax.random.normal(key_, shape=us.shape)


# Define the Feynman--Kac model and the SMC sampler. See Equation (4.4).
def m0(key_):
    return ref_sampler(key_, n=nparticles)


def log_g0(us):
    return log_lk(us, ts[0])


# def log_lk(us, t_k):
#     return jax.vmap(logpdf_likelihood, in_axes=[None, 0])(y, us)


def log_lk(us, t_k):
    return jax.vmap(proxy_log_likelihood, in_axes=[0, None])(us, t_k)


def m(key_, us, tree_param):
    t_km1 = tree_param[0]
    return rev_transition_sampler(key_, us, t_km1)


def log_g(us_k, us_km1, tree_param):
    t_km1, t_k = tree_param
    return log_lk(us_k, t_k) - log_lk(us_km1, t_km1)


# Do conditional sampling
y = 4
nparticles = 100

# samples usT, weights log_wsT, and effective sample sizes esss
key, subkey = jax.random.split(key)
usT, log_wsT, esss = smc_feynman_kac(subkey, m0, log_g0, m, log_g, (ts[:-1], ts[1:]), nparticles, nsteps, stratified, .7, False)

key, subkey = jax.random.split(key)
cond_samples = usT[stratified(subkey, jnp.exp(log_wsT))]

# Plot now
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'text.latex.preamble': r'\usepackage{amsmath,amsfonts}',
    'font.size': 16})

# Plot the truth

xlb, xub = -3, 3
ylb, yub = -3, 3

grid = jnp.linspace(-3, 3, 1000)
meshgrid = jnp.meshgrid(grid, grid)
cartesian = jnp.dstack(meshgrid)

posterior_pdfs = crescent.pdf_x_cond_y(cartesian, y)
plt.contourf(*meshgrid, posterior_pdfs, cmap=plt.cm.binary)
plt.scatter(cond_samples[:, 0], cond_samples[:, 1], s=1, c='tab:blue', alpha=0.2)

plt.text(1, 1, f'$y={y}$')

plt.grid(linestyle='--', alpha=0.3, which='both')
plt.xlim(xlb, xub)
plt.ylim(ylb, yub)
plt.title('Conditional samples by Feynman--Kac')

plt.tight_layout(pad=0.1)
plt.show()

plt.plot(esss)
plt.show()
