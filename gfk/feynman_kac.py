"""
Generic Feynman--Kac models.
"""
import math
import jax
import jax.numpy as jnp
from gfk.likelihoods import pushfwd_normal, pushfwd_normal_batch, bridge_log_likelihood
from gfk.tools import nconcat, euler_maruyama, logpdf_mvn_chol, chol_solve
from gfk.typings import JArray, JKey, PyTree, FloatScalar, NumericScalar, Array
from functools import partial
from typing import Callable, Tuple, Union


def smc_feynman_kac(key: JKey,
                    m0: Union[Callable[[JArray], JArray], JArray],
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
    m0 : Callable [JKey -> (s, ...)] or Array [(s, ...)]
        The initial sampler that draws s independent samples. Or, it can also be an array of samples.
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

    if callable(m0):
        samples0 = m0(key_init)
    else:
        samples0 = m0
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


def _make_euler_disc(rev_drift, rev_dispersion):
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
                              mode: str = 'guided') -> Tuple[Callable, Callable]:
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

    r, rev_c = _make_euler_disc(rev_drift, rev_dispersion)

    def trans_var(i):
        return rev_dispersion(ts[::-1][i]) ** 2 * (ts[i] - ts[i - 1])

    def _markov_common_mean(u, tree_param):
        v_km1, v_k, t_km1, t_k, k, chol_g = tree_param
        inc = r(u, t_km1, t_k)
        rev_c_ = rev_c(t_km1, t_k)
        mean_ = inc + rev_c_ * rev_obs_ops[k].T @ chol_solve(chol_g, v_k - rev_obs_ops[k] @ inc)
        return mean_

    def _markov_common_cov(tree_param):
        _, _, t_km1, t_k, k, chol_g = tree_param
        rev_obs_op = rev_obs_ops[k]
        rev_c_ = rev_c(t_km1, t_k)
        cov_ = rev_c_ * jnp.eye(rev_obs_op.shape[1]) - rev_c_ * rev_obs_op.T @ chol_solve(chol_g, rev_obs_op * rev_c_)
        return cov_

    def m(key, us, tree_param):
        mean_, cov_ = jax.vmap(_markov_common_mean,
                               in_axes=[0, None])(us, tree_param), _markov_common_cov(tree_param)
        return mean_ + jax.random.normal(key, us.shape) @ jnp.linalg.cholesky(cov_).T

    def log_lk(v_k, u_k, k):
        return jax.scipy.stats.multivariate_normal.logpdf(v_k, rev_obs_ops[k] @ u_k, rev_obs_covs[k])

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        v_km1, v_k, t_km1, t_k, k, chol_g = tree_param
        inc = r(u_km1, t_km1, t_k)
        normalising_const = logpdf_mvn_chol(v_k, rev_obs_ops[k] @ inc, chol_g)
        return normalising_const - log_lk(v_km1, u_km1, k - 1)

    rev_obs_ops, rev_obs_covs = pushfwd_normal_batch(obs_op, obs_cov, aux_trans_op, aux_trans_var, trans_var,
                                                     nsteps, reverse=True)
    chols = jax.vmap(lambda rev_obs_op, rev_obs_cov, t_km1, t_k: jnp.linalg.cholesky(
        rev_obs_op @ rev_obs_op.T * rev_c(t_km1, t_k) + rev_obs_cov), in_axes=(0, 0, 0, 0))(rev_obs_ops[1:],
                                                                                            rev_obs_covs[1:],
                                                                                            ts[:-1], ts[1:])

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
        v_km1, v_k, t_km1, t_k, k, _ = tree_param
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
        return guided_smc, log_lk
    elif mode == 'bootstrap':
        return bootstrap_smc, log_lk
    else:
        raise ValueError('Invalid mode.')


def make_fk_seq_lin(logpdf_likelihood: Callable[[Array, Array], FloatScalar],
                    rev_drift, rev_dispersion,
                    aux_inv_semigroup,
                    ts,
                    mode: str = 'guided') -> Callable:
    """Generate an SMC sampler corresponding to a Feynman--Kac model with a Gaussian likelihood.

    Parameters
    ----------
    logpdf_likelihood : Callable (dy, dx) -> float
        The target log-likelihood function.
    rev_drift : Callable (..., float) -> (...)
        The reversal drift function.
    rev_dispersion : Callable (float) -> float
        The reversal dispersion function.
    aux_inv_semigroup : Callable (Int) -> float
        The auxiliary diffusion's inverse semigroup.
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

    r, rev_c = _make_euler_disc(rev_drift, rev_dispersion)

    def propagate(u_k, k):
        def body(i, val):
            return val + rev_drift(val, ts[i]) * (ts[i + 1] - ts[i])

        return jax.lax.fori_loop(k, nsteps, body, u_k)

    def logpdf_rev_transition(u_k, u_km1, t_k, t_km1):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, r(u_km1, t_km1, t_k), rev_c(t_km1, t_k) ** 0.5), axis=-1)

    def _cond_rev_drift(u, v, t, k):
        # Note that the reverse-mode autodiff may not apply here due to a dynamic-size loop.
        # return rev_drift(u, t) + rev_dispersion(t) ** 2 * jax.grad(log_lk, argnums=1)(v, u, k)
        return rev_drift(u, t) + rev_dispersion(t) ** 2 * jax.jacfwd(log_lk, argnums=1)(v, u, k)

    def m(key, us, tree_param):
        _, v_k, t_km1, t_k, k = tree_param
        mean_ = us + jax.vmap(_cond_rev_drift, in_axes=[0, None, None, None])(us, v_k, t_km1, k) * (t_k - t_km1)
        return mean_ + jax.random.normal(key, us.shape) * (t_k - t_km1) ** 0.5

    def logpdf_m(u_k, u_km1, tree_param):
        _, v_k, t_km1, t_k, k = tree_param
        mean_ = u_km1 + _cond_rev_drift(u_km1, v_k, t_km1, k) * (t_k - t_km1)
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, mean_, (t_k - t_km1) ** 0.5))

    def log_lk(v_k, u_k, k):
        return logpdf_likelihood(aux_inv_semigroup(k) * v_k, propagate(u_k, k))

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        v_km1, v_k, t_km1, t_k, k = tree_param
        return (log_lk(v_k, u_k, k) + logpdf_rev_transition(u_k, u_km1, t_k, t_km1)
                - log_lk(v_km1, u_km1, k - 1) - logpdf_m(u_k, u_km1, tree_param))

    def guided_smc(key, m0, vs, nparticles, resampling, resampling_threshold, return_path):

        @partial(jax.vmap, in_axes=[0])
        def log_g0(us):
            return log_lk(vs[0], us, 0)

        return smc_feynman_kac(key, m0, log_g0, m, log_g,
                               (vs[:-1], vs[1:], ts[:-1], ts[1:], jnp.arange(1, nsteps + 1)),
                               nparticles, nsteps, resampling, resampling_threshold,
                               return_path)

    def bs_m(key, us, tree_param):
        _, _, t_km1, t_k, _ = tree_param
        cond_ms = jax.vmap(r, in_axes=[0, None, None])(us, t_km1, t_k)
        cond_scale = rev_c(t_km1, t_k) ** 0.5
        return cond_ms + cond_scale * jax.random.normal(key, shape=us.shape)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def bs_log_g(u_k, u_km1, tree_param):
        v_km1, v_k, _, _, k = tree_param
        return log_lk(v_k, u_k, k) - log_lk(v_km1, u_km1, k - 1)

    def bootstrap_smc(key, m0, vs, nparticles, resampling, resampling_threshold, return_path):

        @partial(jax.vmap, in_axes=[0])
        def log_g0(us):
            return log_lk(vs[0], us, 0)

        return smc_feynman_kac(key, m0, log_g0, bs_m, bs_log_g,
                               (vs[:-1], vs[1:], ts[:-1], ts[1:], jnp.arange(1, nsteps + 1)),
                               nparticles, nsteps, resampling, resampling_threshold,
                               return_path)

    if mode == 'guided':
        return guided_smc
    elif mode == 'bootstrap':
        return bootstrap_smc
    else:
        raise ValueError('Invalid mode.')


def make_fk_bridge(ll_target: Callable[[Array, Array], FloatScalar],
                   ll_ref: Callable[[Array, Array], FloatScalar],
                   alpha: Callable[[FloatScalar], FloatScalar],
                   rev_drift, rev_dispersion,
                   ts,
                   mode: str = 'guided') -> Callable:
    """Generate an SMC sampler corresponding to a Feynman--Kac model with a Gaussian likelihood.

    Parameters
    ----------
    ll_target : Callable (dy, dx) -> float
        The target log-likelihood function.
    ll_ref : Callable (dy, dx) -> float
        The reference log-likelihood function.
    alpha : float -> float
        A function that computes the interpolation parameter. This we use time.
    rev_drift : Callable (..., float) -> (...)
        The reversal drift function.
    rev_dispersion : Callable (float) -> float
        The reversal dispersion function.
    ts : Array (N + 1)
        The times t_0, t_1, ..., t_N.
    langevin_step_size : float
        The step size of the Langevin proposal.
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

    r, rev_c = _make_euler_disc(rev_drift, rev_dispersion)

    def logpdf_rev_transition(u_k, u_km1, t_k, t_km1):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, r(u_km1, t_km1, t_k), rev_c(t_km1, t_k) ** 0.5), axis=-1)

    def _langevin_drift(u_k, u_km1, t_k, t_km1, v_k):
        return 0.5 * (jax.grad(logpdf_rev_transition,
                               argnums=0)(u_k, u_km1, t_k, t_km1) + jax.grad(log_lk, argnums=1)(v_k, u_k, t_k))

    def _cond_rev_drift(u, v, t):
        return rev_drift(u, t) + rev_dispersion(t) ** 2 * jax.grad(log_lk, argnums=1)(v, u, t)

    # def m(key, us, tree_param):
    #     _, v_k, t_km1, t_k = tree_param
    #     mean_ = us + langevin_step_size * jax.vmap(_langevin_drift,
    #                                                in_axes=[0, 0, None, None, None])(us, us, t_k, t_km1, v_k)
    #     return mean_ + jax.random.normal(key, us.shape) * langevin_step_size ** 0.5
    #
    # def logpdf_m(u_k, u_km1, tree_param):
    #     _, v_k, t_km1, t_k = tree_param
    #     mean_ = u_km1 + langevin_step_size * _langevin_drift(u_km1, u_km1, t_k, t_km1, v_k)
    #     return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, mean_, langevin_step_size ** 0.5))
    def m(key, us, tree_param):
        _, v_k, t_km1, t_k = tree_param
        mean_ = us + jax.vmap(_cond_rev_drift, in_axes=[0, None, None])(us, v_k, t_km1) * (t_k - t_km1)
        return mean_ + jax.random.normal(key, us.shape) * (t_k - t_km1) ** 0.5

    def logpdf_m(u_k, u_km1, tree_param):
        _, v_k, t_km1, t_k = tree_param
        mean_ = u_km1 + _cond_rev_drift(u_km1, v_k, t_km1) * (t_k - t_km1)
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, mean_, (t_k - t_km1) ** 0.5))

    def log_lk(v_k, u_k, t_k):
        return bridge_log_likelihood(ll_ref, ll_target, alpha, log=False)(v_k, u_k, t_k)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        v_km1, v_k, t_km1, t_k = tree_param
        return (log_lk(v_k, u_k, t_k) + logpdf_rev_transition(u_k, u_km1, t_k, t_km1)
                - log_lk(v_km1, u_km1, t_km1) - logpdf_m(u_k, u_km1, tree_param))

    def guided_smc(key, m0, vs, nparticles, resampling, resampling_threshold, return_path):

        @partial(jax.vmap, in_axes=[0])
        def log_g0(us):
            return log_lk(vs[0], us, ts[0])

        return smc_feynman_kac(key, m0, log_g0, m, log_g,
                               (vs[:-1], vs[1:], ts[:-1], ts[1:]),
                               nparticles, nsteps, resampling, resampling_threshold,
                               return_path)

    def bs_m(key, us, tree_param):
        _, _, t_km1, t_k = tree_param
        cond_ms = jax.vmap(r, in_axes=[0, None, None])(us, t_km1, t_k)
        cond_scale = rev_c(t_km1, t_k) ** 0.5
        return cond_ms + cond_scale * jax.random.normal(key, shape=us.shape)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def bs_log_g(u_k, u_km1, tree_param):
        v_km1, v_k, t_km1, t_k = tree_param
        return log_lk(v_k, u_k, t_k) - log_lk(v_km1, u_km1, t_km1)

    def bootstrap_smc(key, m0, vs, nparticles, resampling, resampling_threshold, return_path):

        @partial(jax.vmap, in_axes=[0])
        def log_g0(us):
            return log_lk(vs[0], us, ts[0])

        return smc_feynman_kac(key, m0, log_g0, bs_m, bs_log_g,
                               (vs[:-1], vs[1:], ts[:-1], ts[1:]),
                               nparticles, nsteps, resampling, resampling_threshold,
                               return_path)

    if mode == 'guided':
        return guided_smc
    elif mode == 'bootstrap':
        return bootstrap_smc
    else:
        raise ValueError('Invalid mode.')


def make_fk_wu(ll_target,
               rev_drift, rev_dispersion,
               ts, y, langevin_step_size: float,
               cond_expec: Callable[[JArray, float], JArray],
               mode: str = 'guided',
               proposal: str = 'direct',
               bypass_smc: bool = False):
    """Generate the Feynamn--Kac model for Wu's construction.
    """
    nsteps, T = ts.shape[0] - 1, ts[-1]
    r, rev_c = _make_euler_disc(rev_drift, rev_dispersion)

    # def sample_terminal_euler(u, t):
    #     # This is not Tweedie.
    #     return u + rev_drift(u, t) * (T - t)

    def logpdf_rev_transition(u_k, u_km1, t_k, t_km1):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, r(u_km1, t_km1, t_k), rev_c(t_km1, t_k) ** 0.5), axis=-1)

    def _langevin_drift(u_k, u_km1, t_k, t_km1):
        return 0.5 * (jax.grad(logpdf_rev_transition,
                               argnums=0)(u_k, u_km1, t_k, t_km1) + jax.grad(log_lk,
                                                                             argnums=0)(u_k, t_k))

    def m_langevin(key, us, tree_param):
        t_km1, t_k = tree_param
        mean_ = us + langevin_step_size * jax.vmap(_langevin_drift, in_axes=[0, 0, None, None])(us, us, t_k, t_km1)
        return mean_ + jax.random.normal(key, us.shape) * langevin_step_size ** 0.5

    def logpdf_m_langevin(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        mean_ = u_km1 + langevin_step_size * _langevin_drift(u_km1, u_km1, t_k, t_km1)
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, mean_, langevin_step_size ** 0.5))

    def _cond_rev_drift(u, t):
        return rev_drift(u, t) + rev_dispersion(t) ** 2 * jax.grad(log_lk, argnums=0)(u, t)

    def m_direct(key, us, tree_param):
        t_km1, t_k = tree_param
        mean_ = us + jax.vmap(_cond_rev_drift, in_axes=[0, None])(us, t_km1) * (t_k - t_km1)
        scale_ = rev_c(t_km1, t_k) ** 0.5
        return mean_ + jax.random.normal(key, us.shape) * scale_

    def logpdf_m_direct(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        mean_ = u_km1 + _cond_rev_drift(u_km1, t_km1) * (t_k - t_km1)
        scale_ = rev_c(t_km1, t_k) ** 0.5
        return jnp.sum(jax.scipy.stats.norm.logpdf(u_k, mean_, scale_))

    m = m_langevin if proposal == 'langevin' else m_direct
    logpdf_m = logpdf_m_langevin if proposal == 'langevin' else logpdf_m_direct

    def log_lk(u_k, t_k):
        u_N = cond_expec(u_k, t_k)
        return ll_target(y, u_N)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        if bypass_smc:
            return 0.
        else:
            return (log_lk(u_k, t_k) + logpdf_rev_transition(u_k, u_km1, t_k, t_km1)
                    - log_lk(u_km1, t_km1) - logpdf_m(u_k, u_km1, tree_param))

    @partial(jax.vmap, in_axes=[0])
    def log_g0(us):
        return 0. if bypass_smc else log_lk(us, ts[0])

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


def make_mcgdiff(obs_op: Array, obs_cov: Array,
                 rev_drift: Callable, rev_dispersion: Callable,
                 alpha: Callable,
                 y: Array, ts_smc: Array, ts_is: Array, kappa: float,
                 mode: str = 'guided',
                 resample_tau: bool = False,
                 full_final: bool = False):
    """This deals with Y = H X + eps,  eps ~ N(0, v), where v = L L^T.
    This converts to a problem on an inpainting basis. Specifically,
        L^-1 Y = L^-1 H X + N(0, I)
               = U S bar{V}^T X + N(0, I),
               = U S bar{Z} + N(0, I),   Z = V^T X
        U^T L^-1 Y = S bar{Z} + N(0, I),

    Parameters
    ----------
    obs_op : Array (dy, dx)
        The observation operator.
    obs_cov : Array (dy, dy)
        The observation covariance.
    rev_drift : Callable (..., float) -> (...)
        The reversal drift function.
    rev_dispersion : Callable (float) -> float
        The reversal dispersion function.
    alpha : Callable (float) -> float
        The forward process semigroup.
    y : Array (dy, )
        The observation.
    ts_smc : Array (j + 1, )
        The time points for running the noiseless SMC.
    ts_is : Array (k + 1, )
        The time points for running the additional importance sampling. j + k = nsteps
    kappa : float
        The kappa parameter.
    mode : str, default='guided'
        The mode of the SMC sampler.
    resample_tau : bool, default=False
        Whether to resample at t = tau.
    full_final : bool, default=False
        Whether to use an importance sampling between t = tau and t = T.

    Notes
    -----
    Tau must be fixed in order to use JIT. To do so, assume that ts is fixed.
    """
    ts = jnp.concatenate([ts_smc, ts_is[1:]])
    nsteps = ts.shape[0] - 1
    dy, dx = obs_op.shape
    chol = jnp.linalg.cholesky(obs_cov)
    U, S, VT = jnp.linalg.svd(jax.lax.linalg.triangular_solve(chol, obs_op, lower=True, left_side=True),
                              full_matrices=True)
    scaled_y = U.T @ jax.lax.linalg.triangular_solve(chol, y, lower=True, left_side=True)
    inpaint_obs_op = jnp.diag(S)
    inpaint_obs_op_inv = jnp.diag(1 / S)

    # Compute tau with fixed N(0, I)
    tau = ts_smc[-1]
    tau_ind = ts_smc.shape[0] - 1
    h = (1 - kappa) / alpha(tau) ** 2

    if tau_ind > nsteps - 2:
        raise ValueError(f'tau {tau} is too large.')

    # Run for noiseless MCGDiff + one-step importance sampling
    def inpainting_rev_drift(x, t):
        return VT @ rev_drift(VT.T @ x, t)

    def smc_sampler(key, m0, nparticles, resampling, resampling_threshold, return_path):
        key_smc, key_is = jax.random.split(key)
        samples, log_wss, esss = _inpainting_mcgdiff(key_smc, m0,
                                                     inpainting_rev_drift, rev_dispersion, alpha, ts_smc,
                                                     scaled_y,
                                                     inpaint_obs_op_inv, h,
                                                     nparticles, resampling, resampling_threshold, return_path, mode)

        _, subkey = jax.random.split(key_smc)
        if return_path:
            if resample_tau:
                inds = resampling(subkey, jnp.exp(log_wss[-1]))
                samples = samples.at[-1].set(samples[-1, inds])
                log_wss = log_wss.at[-1].set(-jnp.ones(nparticles) * math.log(nparticles))
            us_tau = samples[-1]
            log_ws_tau = log_wss[-1]
        else:
            if resample_tau:
                inds = resampling(subkey, jnp.exp(log_wss))
                samples = samples[inds]
                log_wss = -jnp.ones(nparticles) * math.log(nparticles)
            us_tau = samples
            log_ws_tau = log_wss
        ess_tau = compute_ess(log_ws_tau)

        keys = jax.random.split(key_is, num=nparticles)
        _em = lambda key_, u_: euler_maruyama(key_, u_, ts_is, inpainting_rev_drift, rev_dispersion,
                                              integration_nsteps=1, return_path=return_path)
        uss = jax.vmap(_em, in_axes=[0, 0], out_axes=1)(keys, us_tau)
        usT = uss[-1] if return_path else uss.T

        # The final step
        if not full_final:
            if return_path:
                samples = jnp.concatenate([samples, uss[1:]], axis=0)
                log_wss = jnp.concatenate([log_wss, log_ws_tau * jnp.ones((nsteps - tau_ind, nparticles))], axis=0)
            else:
                samples = usT
                log_wss = log_ws_tau
            esss = jnp.concatenate([esss, ess_tau * jnp.ones(nsteps - tau_ind)])
        else:
            @partial(jax.vmap, in_axes=[0, None])
            def log_lk(u, t):
                return jax.lax.cond(t == ts[-1],
                                    lambda _: jnp.sum(
                                        jax.scipy.stats.norm.logpdf(scaled_y, inpaint_obs_op @ u[:dy], 1.)),
                                    lambda _: jnp.sum(jax.scipy.stats.norm.logpdf(u[:dy],
                                                                                  alpha(
                                                                                      t) * inpaint_obs_op_inv @ scaled_y,
                                                                                  (1 - h * alpha(t) ** 2) ** 0.5)),
                                    None)

            log_wsT = log_ws_tau + log_lk(usT, ts[-1]) - log_lk(us_tau, tau)
            log_wsT = log_wsT - jax.scipy.special.logsumexp(log_wsT)
            essT = compute_ess(log_wsT)

            if return_path:
                samples = jnp.concatenate([samples, uss[1:]], axis=0)
                log_wss = jnp.concatenate([log_wss,
                                           log_ws_tau * jnp.ones((nsteps - 1 - tau_ind, nparticles)),
                                           log_wsT * jnp.ones((1, nparticles))],
                                          axis=0)
            else:
                samples = usT
                log_wss = log_wsT

            esss = jnp.concatenate([esss, ess_tau * jnp.ones(nsteps - 1 - tau_ind), essT * jnp.ones(1)])

        return jnp.einsum('ji,...j->...i', VT, samples), log_wss, esss

    return smc_sampler


def make_noiseless_mcgdiff(obs_op: Array,
                           rev_drift: Callable, rev_dispersion: Callable,
                           alpha: Callable,
                           y: Array, ts: Array):
    U, S, VT = jnp.linalg.svd(obs_op, full_matrices=True)
    inpaint_obs_op = U @ S
    inpaint_obs_op_inv = jnp.diag(1 / jnp.diag(S)) @ U.T

    def inpainting_rev_drift(z, t):
        return VT @ rev_drift(VT.T @ z, t)

    def smc_sampler(key, m0, nparticles, resampling, resampling_threshold):
        key_smc, key_is = jax.random.split(key)
        samples, log_wss, esss = _inpainting_mcgdiff(key_smc, m0,
                                                     inpainting_rev_drift, rev_dispersion, alpha, ts,
                                                     y,
                                                     inpaint_obs_op_inv, 1.,
                                                     nparticles, resampling, resampling_threshold)
        return jnp.einsum('ji,...j->...i', VT, samples), log_wss, esss

    return smc_sampler


def _inpainting_mcgdiff(key, m0,
                        rev_drift, rev_dispersion, alpha: Callable,
                        ts,
                        y: JArray, obs_op_inv, h,
                        nparticles, resampling, resampling_threshold) -> Tuple[JArray, JArray, JArray]:
    """This deals with Y = H bar{X} using MCGDiff, and assume that H has a pseudo-inverse.

    Notes
    -----
    This is an extension of the original MCGDiff that accepts an operator on bar{X}.

    alpha here is different in the MCGDiff paper. Precisely, alpha here operates on the reverse time, and we use a
    square to keep consistent with other methods, i.e., sqrt{original alpha} = this alpha reversed.
    """
    nsteps = ts.shape[0] - 1
    dy = y.shape[0]
    r, rev_c = _make_euler_disc(rev_drift, rev_dispersion)
    inv_y = obs_op_inv @ y

    def log_lk(u, t):
        return jnp.sum(jax.scipy.stats.norm.logpdf(u[:dy], alpha(t) * inv_y, (1 - h * alpha(t) ** 2) ** 0.5))

    def m_and_v(u, t_km1, t_k):
        alp = alpha(t_k)
        c = rev_c(t_km1, t_k)
        trans = r(u, t_km1, t_k)

        gain = (1 - h * alp ** 2) / (1 - h * alp ** 2 + c)
        mean_bar = (1 - gain) * alp * inv_y + gain * trans[:dy]
        var_bar = gain * c * jnp.ones(dy)
        mean_ub = trans[dy:]
        var_ub = c * jnp.ones(u.shape[0] - dy)
        return jnp.concatenate([mean_bar, mean_ub], axis=0), jnp.concatenate([var_bar, var_ub], axis=0)

    def m(key_, us_km1, tree_param):
        t_km1, t_k = tree_param
        ms, vs = jax.vmap(m_and_v, in_axes=[0, None, None])(us_km1, t_km1, t_k)
        return ms + vs ** 0.5 * jax.random.normal(key_, us_km1.shape)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def log_g(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        alp = alpha(t_k)
        normalising_const = jnp.sum(jax.scipy.stats.norm.logpdf(alp * inv_y,
                                                                r(u_km1, t_km1, t_k)[:dy],
                                                                (1 - h * alp ** 2 + rev_c(t_km1, t_k)) ** 0.5))
        return normalising_const - log_lk(u_km1, t_km1)

    @partial(jax.vmap, in_axes=[0])
    def log_g0(us):
        return log_lk(us, ts[0])

    def bs_m(key, us, tree_param):
        t_km1, t_k = tree_param
        cond_ms = jax.vmap(r, in_axes=[0, None, None])(us, t_km1, t_k)
        cond_scale = rev_c(t_km1, t_k) ** 0.5
        return cond_ms + cond_scale * jax.random.normal(key, shape=us.shape)

    @partial(jax.vmap, in_axes=[0, 0, None])
    def bs_log_g(u_k, u_km1, tree_param):
        t_km1, t_k = tree_param
        return log_lk(u_k, t_k) - log_lk(u_km1, t_km1)

    return smc_feynman_kac(key, m0, log_g0, m, log_g,
                           (ts[:-1], ts[1:]),
                           nparticles, nsteps, resampling, resampling_threshold, False)


def make_dc():
    pass
