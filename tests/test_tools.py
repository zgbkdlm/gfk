import numpy as np
import jax.numpy as jnp
import numpy.testing as npt
from gfk.tools import nconcat


def test_nconcat():
    np.random.seed(666)
    arr = jnp.asarray(np.random.randn(10, 5, 3))
    a, b = arr[0], arr[1:]
    npt.assert_array_equal(nconcat(a, b), arr)

    a, b = arr[:-1], arr[-1]
    npt.assert_array_equal(nconcat(a, b), arr)
