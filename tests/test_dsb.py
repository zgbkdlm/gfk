import jax
import jax.numpy as jnp
import numpy.testing as npt
from gfk.dsb import gaussian_bw_sb
from gfk.tools import euler_maruyama

jax.config.update('jax_enable_x64', True)


def test_gaussian_sb():
    """Test the Gaussian Schrodinger bridge.
    """
    key = jax.random.PRNGKey(666)

    mu0 = jnp.array([1.5, -1.8])
    cov0 = jnp.array([[1., 0.3],
                      [0.3, 1.5]])
    chol0 = jnp.linalg.cholesky(cov0)
    mu1 = jnp.array([-1., 2.2])
    cov1 = jnp.array([[0.5, -0.2],
                      [-0.2, 0.7]])

    marginal_mean, marginal_cov, drift = gaussian_bw_sb(mu0, cov0, mu1, cov1, sig=1.)

    # Test marginals
    npt.assert_allclose(mu0, marginal_mean(0.), rtol=1e-8)
    npt.assert_allclose(mu1, marginal_mean(1.), rtol=1e-8)

    npt.assert_allclose(cov0, marginal_cov(0.), rtol=1e-8)
    npt.assert_allclose(cov1, marginal_cov(1.), rtol=1e-8)

    # Test drift
    t0 = 0.
    T = 1.
    nsteps = 128
    ts = jnp.linspace(t0, T, nsteps + 1)
    nsamples = 10000

    def dispersion(x, t):
        return 1.

    def terminal_simulator(key_):
        key_x0, key_sde = jax.random.split(key_)
        x0 = mu0 + chol0 @ jax.random.normal(key_x0, (2,))
        return euler_maruyama(key_sde, x0, ts, drift, dispersion, integration_nsteps=8, return_path=False)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, num=nsamples)
    terminal_samples = jax.vmap(terminal_simulator, in_axes=0)(keys)
    approx_m1 = jnp.mean(terminal_samples, axis=0)
    approx_cov1 = jnp.cov(terminal_samples, rowvar=False)

    npt.assert_allclose(mu1, approx_m1, rtol=5e-2)
    npt.assert_allclose(cov1, approx_cov1, rtol=5e-2)

    # Test if the reverse of the reverse is the forward
    def score(x, t):
        mt, covt = marginal_mean(t), marginal_cov(t)
        chol = jax.scipy.linalg.cho_factor(covt)
        return -jax.scipy.linalg.cho_solve(chol, x - mt)

    def reverse_drift(x, t):
        return -drift(x, 1 - t) + score(x, 1 - t)

    def reverse_reverse_drift(x, t):
        return -reverse_drift(x, t) + score(x, t)

    npt.assert_allclose(reverse_reverse_drift(mu0, 0.5), drift(mu0, 0.5))
