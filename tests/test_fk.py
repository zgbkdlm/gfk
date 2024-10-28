import jax
import jax.numpy as jnp
import numpy.testing as npt
from gfk.feynman_kac import compute_ess


def test_ess():
    n = 10
    ws = jnp.ones(n) / n
    npt.assert_array_equal(compute_ess(jnp.log(ws)), n * 1.)

    ws = jnp.zeros(n)
    ws = ws.at[1].set(1.)
    npt.assert_array_equal(compute_ess(jnp.log(ws)), 1.)
