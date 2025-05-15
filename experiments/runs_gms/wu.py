import math
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from ot.sliced import sliced_wasserstein_distance
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import sampling_gm, logpdf_mvn_chol
from gfk.feynman_kac import make_fk_wu
from gfk.resampling import stratified
from gfk.experiments import generate_gm

parser = argparse.ArgumentParser()
parser.add_argument('--id_l', type=int, help='The MC run starting index.')
parser.add_argument('--id_u', type=int, help='The MC run ending index.')
parser.add_argument('--dx', type=int, default=10, help='The x dimension.')
parser.add_argument('--dy', type=int, default=1, help='The y dimension.')
parser.add_argument('--ncomponents', type=int, default=10, help='The number of GM components.')
parser.add_argument('--offset', type=float, default=0., help='The offset that makes the observation an outlier.')
parser.add_argument('--nparticles', type=int, default=2 ** 14, help='The number of particles; '
                                                                    'the same as with the number of samples')
parser.add_argument('--tweedie', action='store_true', help='Tweedie or Euler')
parser.add_argument('--bypass_smc', action='store_true', help='Use standard conditional SDE sampling.')
args = parser.parse_args()

print(f'Running Wu |Tweedie {args.tweedie}| No SMC {args.bypass_smc} | '
      f'(GM experiment with MCs ({args.id_l}-{args.id_u}), dx={args.dx}, dy={args.dy})')
jax.config.update("jax_enable_x64", True)
keys_mc = np.load('rnd_keys.npy')[args.id_l:args.id_u + 1]

# Define the forward process
a, b = -1., math.sqrt(2)

# Times
t0, T = 0., 2.
nsteps = 100
dt = T / nsteps
ts = jnp.linspace(0., T, nsteps + 1)

# Define the SMC conditional sampler
nparticles = args.nparticles
nsamples = nparticles


# The sampler
@jax.jit
def sampler(key_, obs_op_, obs_cov_, y_, init, target):
    ws_, ms_, eigvals_, eigvecs_ = target
    *_, score, rev_drift, rev_dispersion = make_gm_bridge(ws_, ms_, eigvals_, eigvecs_, a, b, t0, T)
    chol = jnp.linalg.cholesky(obs_cov_)

    def cond_expec_tweedie(u, t):
        alp = jnp.exp(a * (T - t) / 2)
        return 1 / alp * (u - (1 - alp ** 2) * score(u, T - t))

    def cond_expec_euler(u, t):
        return u + rev_drift(u, t) * (T - t)

    smc = make_fk_wu(lambda y_, u: logpdf_mvn_chol(y_, obs_op_ @ u, chol),
                     rev_drift, rev_dispersion,
                     ts, y_, dt * 2,
                     cond_expec_tweedie if args.tweedie else cond_expec_euler,
                     mode='guided', proposal='direct', bypass_smc=True if args.bypass_smc else False)
    samples_, log_ws_, esss_ = smc(key_, init, nparticles, stratified, 0.7, False)
    return samples_, log_ws_, esss_


# MC loop
for k, key_mc in enumerate(keys_mc):
    key_model, key_algo = jax.random.split(key_mc)

    # Generate the model and data
    dx, dy = args.dx, args.dy
    key_model, subkey = jax.random.split(key_model)
    ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, args.ncomponents, full_obs_cov=True)
    eigvals, eigvecs = jnp.linalg.eigh(covs)
    wTs, mTs, eigvalTs, *_ = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T)

    # Define the observation operator and the observation covariance
    y_likely = jnp.einsum('ij,kj,k->i', obs_op, ms, ws)
    y = y_likely + args.offset
    posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
    posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)

    # Generate initial (reference) samples
    key_model, subkey = jax.random.split(key_model)
    keys_ = jax.random.split(subkey, num=nparticles)
    init_particles = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, wTs, mTs, eigvalTs, eigvecs)

    # Compute true samples
    _, subkey = jax.random.split(key_model)
    keys_ = jax.random.split(subkey, num=nparticles)
    post_samples = jax.vmap(sampling_gm, in_axes=[0, None, None, None, None])(keys_, posterior_ws, posterior_ms,
                                                                              posterior_eigvals, posterior_eigvecs)

    # Run the SMC sampler
    samples, log_ws, esss = sampler(key_algo, obs_op, obs_cov, y, init_particles, (ws, ms, eigvals, eigvecs))

    # Compute errors
    swd = sliced_wasserstein_distance(samples, post_samples, a=jnp.exp(log_ws), n_projections=1000, p=1)
    print(f'{k} | Sliced Wasserstein distance: {swd}')

    # Save results
    fn_prefix = 'cond-sde' if args.bypass_smc else 'wu' + '-' + 'tweedie' if args.tweedie else 'euler'
    filename = fn_prefix + '-' + ''
    np.savez(f'./results/gms/wu-{k}',
             samples=samples, log_ws=log_ws, esss=esss, post_samples=post_samples, swd=swd)
