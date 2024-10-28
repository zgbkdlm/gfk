import jax.numpy as jnp
from gfk.typings import JArray


def nconcat(a: JArray, b: JArray) -> JArray:
    """Creating a new leading axis on `a` or `b` and then concat.

    If ndim(a) > ndim(b) then create a new leading axis on `b`.
    """
    if a.ndim > b.ndim:
        return jnp.concatenate([a, b[None, ...]], axis=0)
    else:
        return jnp.concatenate([a[None, ...], b], axis=0)
