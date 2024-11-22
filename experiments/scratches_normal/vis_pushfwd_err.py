"""
Visualise the pushforward likelihood error.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gfk.likelihoods import pushfwd_normal, true_pushfwd_normal
from functools import partial

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(312)

t0, T = 0., 10
nsteps = 100
ts = jnp.linspace(t0, T, nsteps + 1)

m0 = jnp.array([1])
v0 = jnp.array([[2.]])

dy = 1
obs_op = jax.random.normal(key, (dy, 1))
key, subkey = jax.random.split(key)
rnd = jax.random.normal(subkey, (dy, dy))
obs_cov = (rnd @ rnd.T + jnp.eye(dy)) * 0.1

a = -0.5

mT = jnp.exp(a * (T - t0)) * m0
vT = jnp.exp(2 * a * (T - t0)) * v0 - (1 - jnp.exp(2 * a * (T - t0))) / (2 * a)


def aux_semigroup(n, m):
    return jnp.exp(a * (ts[n] - ts[m]))


def aux_trans_var(k):
    return -1 / (2 * a) * (1 - jnp.exp(2 * a * (ts[k + 1] - ts[k])))


def rev_trans_var(k):
    return aux_trans_var(k)


offset = 5
y0 = obs_op @ m0 + offset
ys = jnp.exp(a * ts)[:, None] * y0[None, :]


@jax.jit
@partial(jax.vmap, in_axes=[None, 0, None])
def pushfwd_pdf(y, x, t):
    return jnp.exp(true_pushfwd_normal(y, x, t, m0, v0, obs_op, obs_cov))


@jax.jit
@partial(jax.vmap, in_axes=[None, 0, None])
def approx_pushfwd_pdf(y, x, k):
    f, omega = pushfwd_normal(obs_op, obs_cov, aux_semigroup, aux_trans_var, rev_trans_var, k)
    return jnp.exp(jax.scipy.stats.multivariate_normal.logpdf(y, f @ x, omega))


grid_x = jnp.linspace(-20, 0, 1000)

# def temp(x):
#     return jnp.exp(jax.scipy.stats.multivariate_normal.logpdf(y0, obs_op @ x, obs_cov))
#
# plt.plot(grid_x, jax.vmap(temp)(grid_x[:, None]))
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for i in range(0, nsteps + 1, 20):
    true_pdf = pushfwd_pdf(ys[i], grid_x[:, None], ts[i])
    approx_pdf = approx_pushfwd_pdf(ys[i], grid_x[:, None], i)
    ax.plot(xs=grid_x, ys=true_pdf, zs=ts[i], zdir='x', c='black')
    ax.plot(xs=grid_x, ys=approx_pdf, zs=ts[i], zdir='x', c='tab:red')

# plt.plot(grid_x[:, 0], true_pdfs[-1])
# plt.plot(grid_x[:, 0], approx_pdfs[-1])
plt.show()
