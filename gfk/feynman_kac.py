"""
Generic Feynman--Kac models.
"""
import math
import jax
import jax.numpy as jnp
from gfk.likelihoods import pushfwd_normal, pushfwd_normal_batch
from gfk.tools import nconcat
from gfk.typings import JArray, JKey, PyTree
from functools import partial
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
        axis of size `N`. This container will be a tree-parameter input to the transition kernel and the potential function.
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


def _make_bootstrap_tme(ts, log_likelihood, drift, dispersion, order, nparticles):
    """Make a bootstrap Feynman--Kac model.
    """

    # def step_fn(t):
    #     cond_list = [t == 0.] + [(t > lb) & (t <= ub) for (lb, ub) in zip(block_ts[:-1], block_ts[1:])]
    #     func_list = [lambda _: block_ts[1]] + [lambda _, parg=block_t: parg for block_t in block_ts[1:]]
    #     return jnp.piecewise(t, cond_list, func_list)

    pass


def _make_common(rev_drift, rev_dispersion):
    def r(us, t_k, t_kp1):
        return us + rev_drift(us, t_k) * (t_kp1 - t_k)

    def rev_c(t_k, t_kp1):
        """= C(N - k)"""
        return rev_dispersion(t_k) ** 2 * (t_kp1 - t_k)

    return r, rev_c


def make_fk_normal_likelihood(obs_op, obs_cov,
                              rev_drift, rev_dispersion,
                              aux_trans_op, aux_semigroup, aux_trans_var,
                              ts,
                              mode: str = 'guided') -> Callable:
    """Generate an SMC sampler corresponding to a Feynman--Kac model with a Gaussian likelihood.

    Parameters
    ----------
    obs_op : Array (...)
        The likelihood linear operator.
    obs_cov : Array (...)
        The likelihood covariance.
    rev_drift : Callable (..., float) -> (...)
        The reversal drift function.
    rev_dispersion : Callable (float) -> float
        The reversal dispersion function.
    aux_trans_op : Callable (Int) -> float
        The auxiliary diffusion's transition operator.
    aux_semigroup : Callable (Int, Int) -> float
        The auxiliary diffusion's transition semigroup.
    aux_trans_var : Callable (Int) -> float
        The auxiliary diffusion's transition variance.
    ts : Array (N + 1)
        The times t_0, t_1, ..., t_N.
    mode : str, default='guided'
        The mode of the SMC sampler. Currently, only 'guided' is supported.

    Returns
    -------
    Callable
        A constructed SMC sampler.

    Notes
    -----
    For simplicity, we here assume that the aux process and the reversal dispersion are scalar.
    """
    nsteps = ts.shape[0] - 1

    r, rev_c = _make_common(rev_drift, rev_dispersion)

    def rev_trans_var(i):
        return rev_dispersion(ts[nsteps - i]) ** 2 * (ts[i] - ts[i - 1])

    def logpdf_rev_transition(u_k, u_km1, t_k, t_km1):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, r(u_km1, t_km1, t_k), rev_c(t_km1, t_k) ** 0.5), axis=-1)

    def rev_likelihood(k):
        return pushfwd_normal(obs_op, obs_cov, aux_semigroup, aux_trans_var, rev_trans_var, nsteps - k)

    @partial(jax.vmap, in_axes=[0, None])
    def _markov_common(us, tree_param):
        v_k, v_km1, t_km1, t_k, k, chol_g = tree_param
        inc = r(us, t_km1, t_k)
        rev_obs_op, _ = rev_likelihood(k)
        rev_c_ = rev_c(t_km1, t_k)
        mean_ = inc + rev_c_ * rev_obs_op.T @ jax.scipy.linalg.solve_triangular(chol_g, v_k - rev_obs_op @ inc,
                                                                                lower=True)
        cov_ = rev_c_ * jnp.eye(us.shape[0]) - rev_c_ * rev_obs_op.T @ jax.scipy.linalg.solve_triangular(chol_g,
                                                                                                         rev_obs_op * rev_c_,
                                                                                                         lower=True)
        return mean_, cov_

    def _markov_common_mean(u, tree_param):
        v_k, v_km1, t_km1, t_k, k, chol_g = tree_param
        inc = r(u, t_km1, t_k)
        rev_obs_op, _ = rev_likelihood(k)
        rev_c_ = rev_c(t_km1, t_k)
        mean_ = inc + rev_c_ * rev_obs_op.T @ jax.scipy.linalg.solve_triangular(chol_g, v_k - rev_obs_op @ inc,
                                                                                lower=True)
        return mean_

    def _markov_common_cov(tree_param):
        _, _, t_km1, t_k, k, chol_g = tree_param
        rev_obs_op, _ = rev_likelihood(k)
        rev_c_ = rev_c(t_km1, t_k)
        cov_ = rev_c_ * jnp.eye(rev_obs_op.shape[1]) - rev_c_ * rev_obs_op.T @ jax.scipy.linalg.solve_triangular(chol_g,
                                                                                                                 rev_obs_op * rev_c_,
                                                                                                                 lower=True)
        return cov_

    def m(key, us, tree_param):
        mean_, cov_ = jax.vmap(_markov_common_mean,
                               in_axes=[0, None])(us, tree_param), _markov_common_cov(tree_param)
        return mean_ + jax.random.normal(key, us.shape) @ jnp.linalg.cholesky(cov_).T

    def logpdf_m(u_k, u_km1, tree_param):
        mean_, cov_ = _markov_common_mean(u_km1, tree_param), _markov_common_cov(tree_param)
        return jax.scipy.stats.multivariate_normal.logpdf(u_k, mean_, cov_)

    def log_lk(v_k, u_k, k):
        rev_obs_op_, rev_obs_cov_ = rev_likelihood(k)
        return jax.scipy.stats.multivariate_normal.logpdf(v_k, rev_obs_op_ @ u_k, rev_obs_cov_)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        v_k, v_km1, t_km1, t_k, k, _ = tree_param
        return (log_lk(v_k, u_k, k) + logpdf_rev_transition(u_k, u_km1, t_k, t_km1)
                - log_lk(v_km1, u_km1, k - 1) - logpdf_m(u_k, u_km1, tree_param))

    obs_ops, obs_covs = pushfwd_normal_batch(obs_op, obs_cov, aux_trans_op, aux_trans_var, rev_trans_var, nsteps)
    chols = jax.vmap(lambda rev_obs_op, rev_obs_cov, t_km1, t_k: jnp.linalg.cholesky(
        rev_obs_op @ rev_obs_op.T * rev_c(t_km1, t_k) + rev_obs_cov), in_axes=(0, 0, 0, 0))(obs_ops[-2::-1],
                                                                                            obs_covs[-2::-1], ts[:-1],
                                                                                            ts[1:])

    def guided_smc(key, m0, vs, nparticles, resampling, resampling_threshold, return_path):

        @partial(jax.vmap, in_axes=[0])
        def log_g0(us):
            return log_lk(vs[0], us, 0)

        return smc_feynman_kac(key, m0, log_g0, m, log_g,
                               (vs[:-1], vs[1:], ts[:-1], ts[1:], jnp.arange(1, nsteps + 1), chols),
                               nparticles, nsteps, resampling, resampling_threshold,
                               return_path)

    def bs_m(key, us, tree_param):
        _, _, t_km1, t_k, _, _ = tree_param
        cond_ms = jax.vmap(r, in_axes=[0, None, None])(us, t_km1, t_k)
        cond_scale = rev_c(t_km1, t_k) ** 0.5
        return cond_ms + cond_scale * jax.random.normal(key, shape=us.shape)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def bs_log_g(u_k, u_km1, tree_param):
        v_k, v_km1, t_km1, t_k, k, _ = tree_param
        return log_lk(v_k, u_k, k) - log_lk(v_km1, u_km1, k - 1)

    def bootstrap_smc(key, m0, vs, nparticles, resampling, resampling_threshold, return_path):

        @partial(jax.vmap, in_axes=[0])
        def log_g0(us):
            return log_lk(vs[0], us, 0)

        return smc_feynman_kac(key, m0, log_g0, bs_m, bs_log_g,
                               (vs[:-1], vs[1:], ts[:-1], ts[1:], jnp.arange(1, nsteps + 1), chols),
                               nparticles, nsteps, resampling, resampling_threshold,
                               return_path)

    if mode == 'guided':
        return guided_smc
    elif mode == 'bootstrap':
        return bootstrap_smc
    else:
        raise ValueError('Invalid mode.')


def make_fk_wu_normal(obs_op, obs_cov,
                      rev_drift, rev_dispersion,
                      ts, y, langevin_step_size: float,
                      mode):
    """Generate the Feynamn--Kac model for Wu's construction.
    """
    nsteps, T = ts.shape[0] - 1, ts[-1]
    r, rev_c = _make_common(rev_drift, rev_dispersion)

    def sample_terminal_euler(u, t):
        return u + rev_drift(u, t) * (T - t)

    def sample_terminal_tweedie(u, t):
        pass

    def logpdf_rev_transition(u_k, u_km1, t_k, t_km1):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, r(u_km1, t_km1, t_k), rev_c(t_km1, t_k) ** 0.5), axis=-1)

    def _langevin_drift(u_k, u_km1, t_k, t_km1):
        return 0.5 * (jax.grad(logpdf_rev_transition,
                               argnums=0)(u_k, u_km1, t_k, t_km1) + jax.grad(log_lk,
                                                                             argnums=0)(u_k, t_k))

    def m(key, us, tree_param):
        t_km1, t_k = tree_param
        mean_ = us + langevin_step_size * jax.vmap(_langevin_drift, in_axes=[0, 0, None, None])(us, us, t_k, t_km1)
        return mean_ + jax.random.normal(key, us.shape) * langevin_step_size ** 0.5

    def logpdf_m(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        mean_ = u_km1 + langevin_step_size * _langevin_drift(u_km1, u_km1, t_k, t_km1)
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, mean_, langevin_step_size ** 0.5))

    def log_lk(u_k, t_k):
        u_N = sample_terminal_euler(u_k, t_k)
        return jax.scipy.stats.multivariate_normal.logpdf(y, obs_op @ u_N, obs_cov)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        return (log_lk(u_k, t_k) + logpdf_rev_transition(u_k, u_km1, t_k, t_km1)
                - log_lk(u_km1, t_km1) - logpdf_m(u_k, u_km1, tree_param))

    @partial(jax.vmap, in_axes=[0])
    def log_g0(us):
        return log_lk(us, ts[0])

    def guided_smc(key, m0, nparticles, resampling, resampling_threshold, return_path):
        return smc_feynman_kac(key, m0, log_g0, m, log_g,
                               (ts[:-1], ts[1:]),
                               nparticles, nsteps, resampling, resampling_threshold, return_path)

    def bs_m(key, us, tree_param):
        t_km1, t_k = tree_param
        cond_ms = jax.vmap(r, in_axes=[0, None, None])(us, t_km1, t_k)
        cond_scale = rev_c(t_km1, t_k) ** 0.5
        return cond_ms + cond_scale * jax.random.normal(key, shape=us.shape)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def bs_log_g(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        return log_lk(u_k, t_k) - log_lk(u_km1, t_km1)

    def bootstrap_smc(key, m0, nparticles, resampling, resampling_threshold, return_path):
        return smc_feynman_kac(key, m0, log_g0, bs_m, bs_log_g,
                               (ts[:-1], ts[1:]),
                               nparticles, nsteps, resampling, resampling_threshold, return_path)

    if mode == 'guided':
        return guided_smc
    elif mode == 'bootstrap':
        return bootstrap_smc
    else:
        raise ValueError('Invalid mode.')
