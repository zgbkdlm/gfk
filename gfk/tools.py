import jax
import math
import jax.numpy as jnp
from gfk.typings import JArray, JKey
from typing import Callable


def nconcat(a: JArray, b: JArray) -> JArray:
    """Creating a new leading axis on `a` or `b` and then concat.

    If ndim(a) > ndim(b) then create a new leading axis on `b`.
    """
    if a.ndim > b.ndim:
        return jnp.concatenate([a, b[None, ...]], axis=0)
    else:
        return jnp.concatenate([a[None, ...], b], axis=0)


def sqrtm(mat: JArray, method: str = 'eigh') -> JArray:
    """Matrix (Hermite) square root.
    """
    if method == 'eigh':
        eigenvals, eigenvecs = jnp.linalg.eigh(mat)
        return eigenvecs @ jnp.diag(jnp.sqrt(eigenvals)) @ eigenvecs.T
    else:
        return jnp.real(jax.scipy.linalg.sqrtm(mat))


def bures(m0, cov0, m1, cov1):
    """The Wasserstein distance between two Gaussians.
    """
    sqrt = sqrtm(cov0)
    A = cov0 + cov1 - 2 * sqrtm(sqrt @ cov1 @ sqrt)
    return jnp.sum((m0 - m1) ** 2) + jnp.trace(A)


def _log_det(chol):
    return 2 * jnp.sum(jnp.log(jnp.abs(jnp.diag(chol))))


def kl(m0, cov0, m1, cov1):
    """KL divergence.
    """
    d = m0.shape[-1]
    chol0 = jax.scipy.linalg.cho_factor(cov0)
    chol1 = jax.scipy.linalg.cho_factor(cov1)
    log_det0 = _log_det(chol0[0])
    log_det1 = _log_det(chol1[0])
    return (jnp.trace(jax.scipy.linalg.cho_solve(chol1, cov0))
            - d + jnp.dot(m1 - m0, jax.scipy.linalg.cho_solve(chol1, m1 - m0))
            + log_det1 - log_det0)


def euler_maruyama(key: JKey, x0: JArray, ts: JArray,
                   drift: Callable, dispersion: Callable,
                   integration_nsteps: int = 1,
                   return_path: bool = False) -> JArray:
    r"""Simulate an SDE using the Euler-Maruyama method.

    Parameters
    ----------
    key : JKey
        JAX random key.
    x0 : JArray (..., )
        Initial value.
    ts : JArray (n + 1, )
        Times :math:`t_0, t_1, \ldots, t_n`.
    drift : Callable (..., ), float -> (..., )
        The drift function.
    dispersion : Callable float -> float
        The dispersion function.
    integration_nsteps : int, default=1
        The number of integration steps between each step.
    return_path : bool, default=False
        Whether return the path or just the terminal value.

    Returns
    -------
    JArray (..., ) or JArray (n + 1, ...)
        The terminal value at :math:`t_n`, or the path at :math:`t_0, \ldots, t_n`.
    """
    keys = jax.random.split(key, num=ts.shape[0] - 1)

    def step(xt, t, t_next, key_):
        def scan_body_(carry, elem):
            x = carry
            rnd, t_ = elem
            x = x + drift(x, t_) * ddt + dispersion(t_) * jnp.sqrt(ddt) * rnd
            return x, None

        ddt = jnp.abs(t_next - t) / integration_nsteps
        rnds = jax.random.normal(key_, (integration_nsteps, *x0.shape))
        return jax.lax.scan(scan_body_, xt, (rnds, jnp.linspace(t, t_next - ddt, integration_nsteps)))[0]

    def scan_body(carry, elem):
        x = carry
        key_, t, t_next = elem

        x = step(x, t, t_next, key_)
        return x, x if return_path else None

    terminal_val, path = jax.lax.scan(scan_body, x0, (keys, ts[:-1], ts[1:]))

    if return_path:
        return nconcat(x0, path)
    else:
        return terminal_val


def sampling_gm(key, ws, ms, eigvals, eigvecs):
    n, d = ws.shape[0], ms.shape[1]
    key_cat, key_nor = jax.random.split(key)

    ind = jax.random.choice(key_cat, n, p=ws)
    return ms[ind] + eigvecs[ind] @ (eigvals[ind] ** 0.5 * jax.random.normal(key_nor, (d,)))


def logpdf_mvn(x, m, eigvecs, eigvals):
    n = m.shape[0]
    res = x - m
    c_ = eigvecs.T @ res
    return -0.5 * (jnp.dot(c_, c_ / eigvals) + jnp.sum(jnp.log(eigvals)) + n * math.log(2 * math.pi))
