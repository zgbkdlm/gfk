import jax
import jax.numpy as jnp
import numpy as np
from gfk.typings import Array, JArray, FloatScalar, NumericScalar, IntScalar
from typing import Callable, Tuple


def pushfwd_normal(obs_op: Array,
                   obs_cov: Array,
                   aux_semigroup: Callable[[IntScalar, IntScalar], FloatScalar],
                   aux_trans_var: Callable[[IntScalar], FloatScalar],
                   rev_trans_var: Callable[[IntScalar], FloatScalar],
                   j: IntScalar) -> Tuple[JArray, JArray]:
    """Pushforward a Gaussian likelihood with an auxiliary diffusion.
    This function implements for scalar diffusion coefficients.

    Parameters
    ----------
    obs_op : Array (...)
        The likelihood linear operator.
    obs_cov : Array (...)
        The likelihood covariance.
    aux_semigroup : Callable (Int, Int) -> float
        The auxiliary diffusion's transition semigroup.
    aux_trans_var : Callable (Int) -> float
        The auxiliary diffusion's transition variance.
    rev_trans_var : Callable (Int) -> float
        The reversal diffusion's transition variance.
    j : int
        The step.

    Returns
    -------
    Tuple[Array, Array]
        The pushforward likelihood linear operator and covariance.
    """
    F = aux_semigroup(j, 0) * obs_op

    def body_c(i, val):
        return val + rev_trans_var(i)

    def body_d(i, val):
        return val + aux_semigroup(j, i + 1) ** 2 * aux_trans_var(i)

    _c = jax.lax.fori_loop(1, j + 1, body_c, 0.)
    _d = jax.lax.fori_loop(0, j - 1, body_d, 0.)

    omega = jax.lax.cond(j == 0,
                         lambda _: obs_cov,
                         lambda _: aux_semigroup(j, 0) ** 2 * (obs_op @ obs_op.T * _c + obs_cov) + _d + aux_trans_var(
                             j - 1),
                         None)
    return F, omega


def pushfwd_normal_batch(obs_op: Array,
                         obs_cov: Array,
                         aux_trans_op: Callable[[IntScalar], FloatScalar],
                         aux_trans_var: Callable[[IntScalar], FloatScalar],
                         rev_trans_var: Callable[[IntScalar], FloatScalar],
                         nsteps: int) -> Tuple[Array, Array]:
    """Pushforward a Gaussian likelihood with an auxiliary diffusion.
    This function implements for scalar diffusion coefficients.

    This is the same as with `pushfwd_normal` but will generate a batch of the pushforward likelihoods.

    Parameters
    ----------
    obs_op : Array (...)
        The likelihood linear operator.
    obs_cov : Array (...)
        The likelihood covariance.
    aux_trans_op : Callable (Int) -> float
        The auxiliary diffusion's transition.
    aux_trans_var : Callable (Int) -> float
        The auxiliary diffusion's transition variance.
    rev_trans_var : Callable (Int) -> float
        The reversal diffusion's transition variance.
    nsteps : int
        The number of steps.

    Returns
    -------
    Tuple[(nsteps + 1, ...), (nsteps + 1, ...)]
        The pushforward likelihood linear operators and covariances.
    """
    Fs = np.zeros((nsteps + 1, *obs_op.shape))
    omegas = np.zeros((nsteps + 1, *obs_cov.shape))

    Fs[0] = obs_op
    omegas[0] = obs_cov
    for i in range(nsteps):
        Fs[i + 1] = aux_trans_op(i) * Fs[i]
        omegas[i + 1] = aux_trans_op(i) ** 2 * (Fs[i] @ Fs[i].T * rev_trans_var(i + 1) + omegas[i]) + aux_trans_var(i)
    return Fs, omegas


def true_pushfwd_normal(y, x, t, m, v, obs_op, obs_cov):
    dy, dx = obs_op.shape
    fwd_op, fwd_var = jnp.exp(-0.5 * t), 1 - jnp.exp(-t)
    my, vy = obs_op @ m, obs_op @ v @ obs_op.T + obs_cov
    cov_xy = v @ obs_op.T

    mk, vk = m * fwd_op, fwd_op ** 2 * v + jnp.eye(dx) * fwd_var
    myk, vyk = my * fwd_op, fwd_op ** 2 * vy + jnp.eye(dy) * fwd_var
    cov_xyk = fwd_op ** 2 * cov_xy
    cov_yxk = cov_xyk.T
    chol = jax.scipy.linalg.cho_factor(vk)

    cond_m = myk + cov_yxk @ jax.scipy.linalg.cho_solve(chol, x - mk)
    cond_v = vyk - cov_yxk @ jax.scipy.linalg.cho_solve(chol, cov_xyk)
    return jax.scipy.stats.multivariate_normal.logpdf(y, cond_m, cond_v)


def bridge_log_likelihood(ref_ll: Callable[[Array, Array], FloatScalar],
                          target_ll: Callable[[Array, Array], FloatScalar],
                          alpha: Callable[[NumericScalar], FloatScalar],
                          log: bool = False) -> Callable[[Array, Array, FloatScalar], FloatScalar]:
    """Make an interpolating proxy log-likelihood between the reference and target log-likelihoods.

    Parameters
    ----------
    ref_ll : Callable (dy, dx) -> float
        The reference log-likelihood function.
    target_ll : Callable (dy, dx) -> float
        The target log-likelihood function.
    alpha : number -> float
        A function that computes the interpolation parameter. Its value should start from 0 and end at 1.
    log : bool, default=True
        Bridging the log pdf or the pdf.

    Returns
    -------
    Callable (dy, dx, float) -> float
        The interpolating proxy log-likelihood function.
    """

    def bridged_ll(y, x, t):
        alp = alpha(t)
        log_ref = ref_ll(y, x)
        log_tar = target_ll(y, x)

        if log:
            return (1 - alp) * log_ref + alp * log_tar
        else:
            bs = jnp.array([1 - alp, alp])
            vs = jnp.array([log_ref, log_tar])
            return jax.scipy.special.logsumexp(a=vs, b=bs)

    return bridged_ll
