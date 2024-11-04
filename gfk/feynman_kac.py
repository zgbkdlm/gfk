"""
Generic Feynman--Kac models.
"""
import math
import jax
import jax.numpy as jnp
from gfk.tools import nconcat
from gfk.typings import JArray, JKey, PyTree
from typing import Callable, Tuple


def smc_feynman_kac(key: JKey,
                    m0: Callable[[JArray], JArray],
                    log_g0: Callable[[JArray], JArray],
                    m: Callable[[JArray, JArray, PyTree], JArray],
                    log_g: Callable[[JArray, JArray, PyTree], JArray],
                    scan_pytree: PyTree,
                    nparticles: int,
                    nsteps: int,
                    resampling: Callable[[JKey, JArray], JArray],
                    resampling_threshold: float = 1.,
                    return_path: bool = False) -> Tuple[JArray, JArray, JArray]:
    r"""Sequential Monte Carlo simulation of a Feynman--Kac model.

    .. math::

        Q_{0:N}(u_{0:N}) \propto M0(u0) \, G0(u0) \prod_{n=1}^N M_{n \mid n-1}(u_n \mid u_{n-1}) \, G_n(u_n, u_{n-1})

    Parameters
    ----------
    key : JKey
        A JAX random key.
    m0 : Callable [JKey -> (s, ...)]
        The initial sampler that draws s independent samples.
    log_g0 : Callable [(s, ...) -> (s, )]
        The initial (log) potential function. Given `s` samples, this function should return `s` weights in an array.
    m : Callable [JKey, (s, ...), PyTree -> (s, ...)]
        The Markov transition kernel at k. From left to right, the arguments are, key, input samples, and a pytree
        parameter. The output is an array of the leading axis of the input samples array.
    log_g : Callable [(s, ...), (s, ...), PyTree -> (s, )]
        The (log) potential function. From left to right, the arguments are for `u_n`, `u_{n-1}`, and a pytree
        parameter. The output is an array of size `s`.
    scan_pytree : PyTree
        A PyTree container that is going to be scanned over scan steps. The elements should have a consistent leading
        axis of size `N`. This container will be an input to the transition kernel and the potential function.
    nparticles : int
        The number of particles `s`.
    nsteps : int
        The number of time steps `N`.
    resampling : Callable [JKey, (s, ) -> (s, )]
        The resampling scheme. Given a pair of JKey and weights, this function should return an array of indices for
        the resampling.
    resampling_threshold : float, default=1.
        The threshold of ESS for resampling. If the current ESS < threshold * N, then apply resampling. Default is 1
        meaning resampling at every step.
    return_path : bool, default=False
        Set True to return all the historical particles.

    Returns
    -------
    JArrays [(N + 1, s, ...), (N + 1, s), (N + 1, )] or [(s, ...), (s, ), (N + 1, )]
    A tuple of three arrays. If `return_path` then the return sizes of the arrays are
    `(N + 1, s, ...), (N + 1, s), (N + 1, )`. Else are (s, ...), (s, ), (N + 1, ).
    """
    key_init, key_body = jax.random.split(key)
    flat_log_ws = -math.log(nparticles) * jnp.ones(nparticles)

    samples0 = m0(key_init)
    log_ws0_ = log_g0(samples0)
    log_ws0 = log_ws0_ - jax.scipy.special.logsumexp(log_ws0_)
    ess0 = compute_ess(log_ws0)

    def scan_body(carry, elem):
        samples, log_ws, ess = carry
        pytree, key_k = elem
        key_resample, key_markov = jax.random.split(key_k)

        samples, log_ws = jax.lax.cond(ess < resampling_threshold * nparticles,
                                       lambda _: (samples[resampling(key_resample, jnp.exp(log_ws))], flat_log_ws),
                                       lambda _: (samples, log_ws),
                                       None)

        prop_samples = m(key_markov, samples, pytree)
        log_ws_ = log_ws + log_g(prop_samples, samples, pytree)
        log_ws = log_ws_ - jax.scipy.special.logsumexp(log_ws_)
        ess = compute_ess(log_ws)

        return (prop_samples, log_ws, ess), (prop_samples, log_ws, ess) if return_path else ess

    keys = jax.random.split(key_body, num=nsteps)
    if return_path:
        _, (sampless, log_wss, esss) = jax.lax.scan(scan_body, (samples0, log_ws0, ess0), (scan_pytree, keys))
        return nconcat(samples0, sampless), nconcat(log_ws0, log_wss), nconcat(ess0, esss)
    else:
        (samplesN, log_wsN, _), esss = jax.lax.scan(scan_body, (samples0, log_ws0, ess0), (scan_pytree, keys))
        return samplesN, log_wsN, nconcat(ess0, esss)


def csmc():
    # TODO
    pass


def compute_ess(log_ws: JArray) -> JArray:
    """Effective sample size.
    """
    return jnp.exp(-jax.scipy.special.logsumexp(log_ws * 2))


def bootstrap_tme():
    # TODO
    pass
