"""
Test the synthetic target distributions used in the experiments.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from gfk.synthetic_targets import make_gsb
from functools import partial

jax.config.update('jax_enable_x64', True)


def test_gsb():
    key = jax.random.PRNGKey(666)
    _, _, m, cov, _, _, log_likelihood, _, _, posterior = make_gsb(key, d=10)
    y = jnp.ones(10)
    computed_posterior_m, computed_posterior_cov = posterior(y)

    def energy(x):
        return log_likelihood(y, x) + jax.scipy.stats.multivariate_normal.logpdf(x, m, cov)

    npt.assert_allclose(jax.grad(energy)(computed_posterior_m), 0.,
                        atol=1e-10)
    npt.assert_allclose(-jnp.linalg.inv(jax.hessian(energy)(computed_posterior_m)), computed_posterior_cov, rtol=1e-10)
