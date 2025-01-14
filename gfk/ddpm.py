import math
import jax.numpy as jnp

from experiments.scratches_gms.aux_bridge import alpha


def ddpm2sde(unet_model, ddpm_scheduler, ts):
    """Given a DDPM model using diffusers, convert it a continuous-time SDE model with Euler-discretised reversal.

    Parameters
    ----------
    unet_model : UNet2DModel
        The DDPM model.
    ddpm_scheduler : DDPMScheduler
        The DDPM scheduler.
    ts : Array (nsteps + 1, )
        The time steps. Assume evenly spaced.
    """
    if ddpm_scheduler.beta_schedule != 'linear':
        raise NotImplementedError('Only the linear beta schedules is implemented.')

    if ddpm_scheduler.prediction_type != 'epsilon':
        raise NotImplementedError('Only the epsilon model is implemented.')

    t0 = ts[0]
    T = ts[-1]
    nsteps = ts.shape[0] - 1
    dt = T / nsteps
    beta_min = ddpm_scheduler.config.beta_start
    beta_max = ddpm_scheduler.config.beta_end

    def beta(t):
        return (beta_max - beta_min) / (T - t0) * t + (beta_min * T - beta_max * t0) / (T - t0)

    def beta_integral(t, s):
        return 0.5 * (t - s) * ((beta_max - beta_min) / (T - t0) * (t + s)
                                + 2 * (beta_min * T - beta_max * t0) / (T - t0))

    def fwd_semigroup(t):
        return jnp.exp(-0.5 * beta_integral(t, t0))

    def fwd_variance(t):
        return 1 - jnp.exp(-beta_integral(t, t0))

    def alpha(t):
        return 1 - beta(t)

    def alpha_bar(t):
        return fwd_semigroup(t) ** 2

    def score(x, t):
        return unet_model(x, t / dt).sample / (1 - alpha_bar(t)) ** 0.5

    def rev_drift(x, t):
        """This corresponds to the DDPM sampling step.
        """
        alp_t = alpha(T - t)
        return ((1 - alp_t ** 0.5) / alp_t ** 0.5 - (1 - alp_t) / alp_t ** 0.5 * score(x, T - t)) / dt

    def rev_dispersion(t):
        return beta(T - t) ** 0.5

    return fwd_semigroup, fwd_variance, rev_drift, rev_dispersion


def ddpm_disc(unet_model, ddpm_scheduler, nsteps):
    """The native DDPM in discrete times k = 1, 2, ..., nsteps + 1.
    """
    beta_min = ddpm_scheduler.config.beta_start
    beta_max = ddpm_scheduler.config.beta_end
    k_min = 1
    k_max = nsteps + 1
    ks = jnp.arange(k_min, k_max + 1)

    def beta(k):
        return (beta_max - beta_min) / (k_max - k_min) * k + (beta_min * k_max - beta_max * k_min) / (k_max - k_min)

    alp_prods = jnp.cumprod(1 - beta(ks))

    def alpha_bar(k):
        return alp_prods[k - 1]

    def fwd_semigroup(k):
        return alpha_bar(k) ** 0.5

    def fwd_variance(k):
        return 1 - alpha_bar(k)

    def rev_transition(x, k):
        k = k_max - k + 1
        alp_k = 1 - beta(k)
        return 1 / alp_k ** 0.5 * (x - (1 - alp_k) / (1 - alpha_bar(k)) ** 0.5 * unet_model(x, k).sample)

    def rev_variance(k):
        k = k_max - k + 1
        return beta(k)

    return fwd_semigroup, fwd_variance, rev_transition, rev_variance
