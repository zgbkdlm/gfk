import pytest
import jax.random
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from gfk.tools import nconcat, sqrtm, kl, bures, logpdf_mvn, logpdf_mvn_chol, chol_solve

jax.config.update("jax_enable_x64", True)


def test_nconcat():
    np.random.seed(666)
    arr = jnp.asarray(np.random.randn(10, 5, 3))
    a, b = arr[0], arr[1:]
    npt.assert_array_equal(nconcat(a, b), arr)

    a, b = arr[:-1], arr[-1]
    npt.assert_array_equal(nconcat(a, b), arr)


@pytest.mark.parametrize("method", ['eigh', 'scipy'])
def test_sqrtm(method):
    key = jax.random.PRNGKey(666)
    a = jax.random.normal(key, (3, 3))
    a = a @ a.T
    mat = a @ a

    npt.assert_allclose(sqrtm(mat, method='eigh'), a)


def test_kl_bures():
    m0, m1 = jnp.ones((2, 10))
    cov0, cov1 = jnp.eye(10), jnp.eye(10)
    npt.assert_allclose(0., kl(m0, cov0, m1, cov1))
    npt.assert_allclose(0., bures(m0, cov0, m1, cov1))


def test_mvns():
    x = jnp.array([1., 2., 3.3])
    m = jnp.array([0.2, 0.1, 0.4])
    cov = jnp.array([[1., 0.1, 0.2],
                     [0.1, 2., 0.1],
                     [0.2, 0.1, 2.5]])
    eigvals, eigvecs = jnp.linalg.eigh(cov)
    chol = jnp.linalg.cholesky(cov)

    npt.assert_allclose(logpdf_mvn(x, m, eigvals, eigvecs), jax.scipy.stats.multivariate_normal.logpdf(x, m, cov))
    npt.assert_allclose(logpdf_mvn_chol(x, m, chol), jax.scipy.stats.multivariate_normal.logpdf(x, m, cov))


def test_chol_solve():
    cov = jnp.array([[1., 0.1, 0.2],
                     [0.1, 2., 0.1],
                     [0.2, 0.1, 2.5]])
    chol = jnp.linalg.cholesky(cov)
    x = jnp.ones(3)

    npt.assert_allclose(chol_solve(chol, x), jnp.linalg.solve(cov, x))
