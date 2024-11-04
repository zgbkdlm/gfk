"""
The test problem in the pedagogical example.
"""
from collections.abc import Callable

import jax
import jax.numpy as jnp
from gfk.dsb import gaussian_bw_sb
from gfk.typings import JArray, JKey, FloatScalar
from typing import Tuple


class Crescent:
    r"""A crescent-shaped posterior distribution.

    X ~ GM(m0, v0, w0, m1, v1, w1)
    Y | X ~ N(Y | X_1 / c + 0.5 * (X_0 ^ 2 + c ^ 2), xi)
    """

    def __init__(self, c: float, xi: float):
        self.c = c
        self.xi = xi

        self.w0 = 0.5
        self.m0 = jnp.array([0., 0.])
        self.v0 = jnp.array([[1., 0.8],
                             [0.8, 1.]])
        self.chol_v0 = jnp.linalg.cholesky(self.v0)

        self.w1 = 1 - self.w0
        self.m1 = -self.m0
        self.v1 = jnp.array([[1., -0.8],
                             [-0.8, 1.]])
        self.chol_v1 = jnp.linalg.cholesky(self.v1)

        self.ws = jnp.array([self.w0, self.w1])
        self.ms = jnp.concatenate([self.m0[None, :], self.m1[None, ::]], axis=0)
        self.vs = jnp.concatenate([self.v0[None, ...], self.v1[None, ...]], axis=0)
        self.chols = jnp.concatenate([self.chol_v0[None, ...], self.chol_v1[None, ...]], axis=0)

    def sampler_x(self, key: JKey):
        key_cat, key_x = jax.random.split(key)
        ind = jax.random.choice(key_cat, 2, p=self.ws)
        return self.ms[ind] + self.chols[ind] @ jax.random.normal(key_x, shape=(2,))

    def emission(self, x):
        return x[..., 1] / self.c + 0.5 * (x[..., 0] ** 2 + self.c ** 2)

    def sampler_y_cond_x(self, key: JKey, x: JArray):
        m = self.emission(x)
        return m + jnp.sqrt(self.xi) * jax.random.normal(key, shape=m.shape)

    def sampler_joint(self, key: JKey) -> Tuple[JArray, JArray]:
        key_x, key_y = jax.random.split(key)
        x = self.sampler_x(key_x)
        y = self.sampler_y_cond_x(key_y, x)
        return x, y

    def logpdf_x(self, x):
        log_pdf0 = jax.scipy.stats.multivariate_normal.logpdf(x, self.ms[0], self.vs[0])
        log_pdf1 = jax.scipy.stats.multivariate_normal.logpdf(x, self.ms[1], self.vs[1])
        return jax.scipy.special.logsumexp(jnp.array([jnp.log(self.w0) + log_pdf0,
                                                      jnp.log(self.w1) + log_pdf1]))

    def logpdf_y_cond_x(self, y, x):
        return jax.scipy.stats.norm.logpdf(y, self.emission(x), jnp.sqrt(self.xi))

    def pdf_x_cond_y(self, xs_mesh: JArray, y: FloatScalar) -> JArray:
        """

        Parameters
        ----------
        xs_mesh : JArray (n1, n2, d)
        y : FloatScalar

        Returns
        -------
        JArray (n1, n2)
            The PDF evaluated at the Cartesian grids.
        """

        def unnormalised_joint(x_):
            return jnp.exp(self.logpdf_y_cond_x(y, x_) + self.logpdf_x(x_))

        evals = jax.vmap(jax.vmap(unnormalised_joint, in_axes=[0]), in_axes=[0])(xs_mesh)
        ell = jax.scipy.integrate.trapezoid(jax.scipy.integrate.trapezoid(evals, xs_mesh[0, :, 0], axis=0),
                                            xs_mesh[:, 0, 1])
        return evals / ell

    def estimate_bound(self, y: FloatScalar):
        """Based on the value y, give a reasonable estimate of the region of the posterior.
        """

        def unnormalised_joint(x_):
            return jnp.exp(self.logpdf_y_cond_x(y, x_) + self.logpdf_x(x_))

        x = jnp.zeros(2)
        cov = -jnp.linalg.inv(jax.hessian(unnormalised_joint)(x))
        return jnp.sqrt(jnp.diagonal(cov)) * 2


class GM:
    """This will give an almost exact model, and the errors comes from asymptotic T and the discretisation.
    """
    pass


def make_gsb(key, d, sig=1.) -> Tuple[JArray, JArray, JArray, JArray, Callable, Callable, Callable, Callable]:
    """This will give an exact model, and the only error comes from the discretisation.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    d : int
        The dimension of the Gaussian distribution.
    sig : float, default=1.
        The Brownian motion's diffusion coefficient.

    Returns
    -------
    JArray, JArray, JArray, JArray, Callable, Callable
        The ref mean, ref covariance, target mean, target covariance, drift, and dispersion functions.
    """
    key, subkey = jax.random.split(key)
    key_m_ref, key_m, key_cov_ref, key_cov, key_ll_c, key_h = jax.random.split(subkey, num=6)

    m_ref = jax.random.normal(key_m_ref, shape=(d,))
    m = jax.random.normal(key_m, shape=(d,))
    g0 = jax.random.normal(key_cov_ref, shape=(d,))
    g1 = jax.random.normal(key_cov, shape=(d,))
    cov_ref = jnp.outer(g0, g0) + jnp.eye(d)
    cov = jnp.outer(g1, g1) + jnp.eye(d)

    h = jax.random.normal(key_h, shape=(d, d))
    z = h @ cov @ h.T + jnp.eye(d)
    chol_z = jax.scipy.linalg.cho_factor(z)

    def dispersion(x, t):
        return 1.

    _, _, drift = gaussian_bw_sb(m_ref, cov_ref, m, cov, sig=sig)

    ll_cs = jax.random.normal(key_ll_c, shape=(d,))

    def log_likelihood(y, x):
        s = d // 2
        x1, x2 = x[:s], x[s:]
        return jax.scipy.stats.norm.logpdf(y, jnp.dot(ll_cs, jnp.concatenate([x1, x2 ** 2])), 1.)

    def log_linear_likelihood(y, x):
        return jax.scipy.stats.multivariate_normal.logpdf(y, h @ x, jnp.eye(d))

    def posterior_linear(y):
        v = cov @ h.T
        return m + v @ jax.scipy.linalg.cho_solve(chol_z, y - h @ m), cov - v @ jax.scipy.linalg.cho_solve(chol_z, v.T)

    def posterior(x, y):
        pass

    return m_ref, cov_ref, m, cov, drift, dispersion, log_linear_likelihood, posterior_linear
