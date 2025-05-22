import math
import jax
import jax.numpy as jnp
import numpy as np
import argparse
from ot.sliced import sliced_wasserstein_distance
from gfk.synthetic_targets import make_gm_bridge, gm_lin_posterior
from gfk.tools import sampling_gm
from gfk.feynman_kac import make_fk_normal_likelihood
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
parser.add_argument('--noiseless', action='store_true', help='Noiseless observations.')
args = parser.parse_args()

print(f'Running {"aux-noiseless" if args.noiseless else "aux"} '
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

# Define the auxiliary process
aux_a, aux_b = a, b


def aux_trans_op(i):
    return jnp.exp(aux_a * (ts[i + 1] - ts[i]))


def aux_semigroup(n, m):
    return jnp.exp(aux_a * (ts[n] - ts[m]))


def aux_trans_var(i):
    return aux_b ** 2 / (2 * aux_a) * (jnp.exp(2 * aux_a * (ts[i + 1] - ts[i])) - 1)


# Define the SMC conditional sampler
nparticles = args.nparticles
nsamples = nparticles


# The sampler
@jax.jit
def sampler(key_, obs_op_, obs_cov_, vs_, init, target):
    ws_, ms_, eigvals_, eigvecs_ = target
    *_, rev_drift, rev_dispersion = make_gm_bridge(ws_, ms_, eigvals_, eigvecs_, a, b, t0, T)
    smc, _ = make_fk_normal_likelihood(obs_op_, obs_cov_, rev_drift, rev_dispersion,
                                       aux_trans_op, aux_semigroup, aux_trans_var,
                                       ts, mode='guided')
    samples_, log_ws_, esss_ = smc(key_, init, vs_, nparticles, stratified, 0.7, False)
    return samples_, log_ws_, esss_


# MC loop
for k, key_mc in enumerate(keys_mc):
    key_model, key_algo = jax.random.split(key_mc)

    # Generate the model and data
    dx, dy = args.dx, args.dy
    key_model, subkey = jax.random.split(key_model)
    ws, ms, covs, obs_op, obs_cov = generate_gm(subkey, dx, dy, args.ncomponents,
                                                full_obs_cov=True, noiseless=args.noiseless)
    eigvals, eigvecs = jnp.linalg.eigh(covs)
    wTs, mTs, eigvalTs, *_ = make_gm_bridge(ws, ms, eigvals, eigvecs, a, b, t0, T)

    # Define the observation operator and the observation covariance
    y_likely = jnp.einsum('ij,kj,k->i', obs_op, ms, ws)
    y = y_likely + args.offset
    posterior_ws, posterior_ms, posterior_covs = gm_lin_posterior(y, obs_op, obs_cov, ws, ms, covs)
    posterior_eigvals, posterior_eigvecs = jnp.linalg.eigh(posterior_covs)

    ys = jax.vmap(lambda n: aux_semigroup(n, 0) * y, in_axes=0)(jnp.arange(nsteps + 1))
    vs = ys[::-1]

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
    samples, log_ws, esss = sampler(key_algo, obs_op, obs_cov, vs, init_particles, (ws, ms, eigvals, eigvecs))

    # Compute errors
    swd = sliced_wasserstein_distance(samples, post_samples, a=jnp.exp(log_ws), n_projections=1000, p=1)
    print(f'{k} | Sliced Wasserstein distance: {swd}')

    # Save results
    fn_prefix = 'aux-noiseless' if args.noiseless else 'aux'
    filename = fn_prefix + f'-{dx}-{nparticles}-{args.offset}-{k}'
    np.savez(f'./results/gms/{filename}',
             samples=samples, log_ws=log_ws, esss=esss, post_samples=post_samples, swd=swd)
