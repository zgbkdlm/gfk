import pytest
import jax.random
import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from gfk.tools import nconcat, sqrtm

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
