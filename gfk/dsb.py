"""
The diffusion Scrödinger bridge implementation.

Based on https://github.com/spdes/gdcs/blob/main/gdcs/dsb.py.
"""
import jax
import jax.numpy as jnp
from gfk.tools import sqrtm
from gfk.typings import JArray, Array, JKey, FloatScalar, JFloat
from typing import Callable, Tuple


def ipf_loss_cont(key: JKey,
                  param: JArray,
                  simulator_param: JArray,
                  init_samples: JArray,
                  ts: JArray,
                  parametric_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  simulator_drift: Callable[[JArray, FloatScalar, JArray], JArray],
                  dispersion: Callable) -> JFloat:
    r"""The iterative proportional fitting (continuous version) loss used in Schrödinger bridge.
    Proposition 29, de Bortoli et al., 2021.

    Parameters
    ----------
    key : JKey
        A JAX random key.
    param : JArray
        The parameter of the parametric drift function that you wish to learn.
    simulator_param : JArray
        The parameter of the simulation process drift function.
    init_samples : JArray (m, n, ...)
        Samples from the initial distribution (i.e., either the target or the reference depending on if you are
        learning the forward or the backward process).
    ts : JArray (n, )
        Either the forward times `t_0, t_1, ..., t_n` or its reversal, depending on if you are using this function
        to learn the backward or forward process.
    parametric_drift : Callable
        The parametric drift function whose signature is `f(x, t, param)`.
    simulator_drift : Callable
        The simulator process' drift function whose signature is `g(x, t, simulator_param)`.
    dispersion : Callable
        The dispersion function, a function of time.

    Returns
    -------
    JFloat
        The loss.

    Notes
    -----
    When using this function to learn the backward process `target <- ref`,
    simulate the forward process defined by `simulator_drift` at forward `ts`.

    When using this function to learn the forward process `target -> ref`,
    simulate the backward process defined by `simulator_drift` at backward `ts`.
    """
    nsteps = ts.shape[0] - 1
    fn = lambda z, t, dt: z + simulator_drift(z, t, simulator_param) * dt

    def scan_body(carry, elem):
        z, err = carry
        t, t_next, rnd = elem

        dt = jnp.abs(t_next - t)
        z_next = z + simulator_drift(z, t, simulator_param) * dt + jnp.sqrt(dt) * dispersion(t) * rnd
        err = err + jnp.mean(
            (parametric_drift(z_next, t_next, param) * dt - (fn(z, t, dt) - fn(z_next, t, dt))) ** 2)
        return (z_next, err), None

    key, subkey = jax.random.split(key)
    rnds = jax.random.normal(subkey, (nsteps, *init_samples.shape))
    (_, err_final), _ = jax.lax.scan(scan_body, (init_samples, 0.), (ts[:-1], ts[1:], rnds))
    return jnp.mean(err_final / nsteps)


def gaussian_bw_sb(mean0: Array, cov0: Array,
                   mean1: Array, cov1: Array,
                   sig: float = 1.) -> Tuple[Callable, Callable, Callable]:
    """Generate a Gaussian Schrödinger bridge with a Brownian motion reference at time interval [0, 1].

    Parameters
    ----------
    mean0 : Array (d, )
        The mean of the initial Gaussian distribution.
    cov0 : Array (d, d)
        The covariance of the initial Gaussian distribution.
    mean1 : Array (d, )
        The mean of the terminal Gaussian distribution.
    cov1 : Array (d, d)
        The covariance of the terminal Gaussian distribution.
    sig : float, default=1.
        The Brownian motion's diffusion coefficient.

    Returns
    -------
    Tuple[Callable, Callable, Callable]
        The marginal mean, marginal covariance, and drift functions.

    Notes
    -----
    Table 1, The Schrödinger Bridge between Gaussian Measures has a Closed Form, 2023.
    """
    d = mean0.shape[0]
    sqrt0 = sqrtm(cov0)

    D_sig = sqrtm(4 * sqrt0 @ cov1 @ sqrt0 + sig ** 4 * jnp.eye(d))
    C_sig = 0.5 * (sqrt0 @ jnp.linalg.solve(sqrt0.T, D_sig.T).T - sig ** 2 * jnp.eye(d))

    def kappa(t, _):
        return t * sig ** 2

    def r(t):
        return t

    def r_bar(t):
        return 1 - t

    def rho(t):
        return t

    def marginal_mean(t):
        return r_bar(t) * mean0 + r(t) * mean1

    def marginal_cov(t):
        return r_bar(t) ** 2 * cov0 + r(t) ** 2 * cov1 + r(t) * r_bar(t) * (C_sig + C_sig.T) + kappa(t, t) * (
                1 - rho(t)) * jnp.eye(d)

    def s(t):
        pt = r(t) * cov1 + r_bar(t) * C_sig
        qt = r_bar(t) * cov0 + r(t) * C_sig
        return pt - qt.T - sig ** 2 * rho(t) * jnp.eye(d)

    def drift(x, t):
        mt = marginal_mean(t)
        chol_t = jax.scipy.linalg.cho_factor(marginal_cov(t))
        return s(t).T @ jax.scipy.linalg.cho_solve(chol_t, x - mt) - mean0 + mean1

    return marginal_mean, marginal_cov, drift
