"""
Neural networks with FLAX.

Implementations based on https://github.com/zgbkdlm/fbs/blob/main/fbs/dsb/base.py.
"""
import math
import flax.linen as nn
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from gfk.typings import JArray, FloatScalar, JKey
from typing import Tuple, Callable, Sequence


def make_nn(key: JKey, neural_network: nn.Module, shape_x: Sequence[int], shape_t: Sequence[int]) -> Tuple[
    JArray, Callable[[JArray], dict], Callable[[JArray, FloatScalar, JArray], JArray]]:
    """Make a neural network for approximating a spatial-temporal function :math:`f(x, t)`.

    Parameters
    ----------
    neural_network : linen.Module
        A neural network instance.
    shape_x : (int, ...)
        The spatial dimension, where the leading axis means the batch size.
    shape_t : (int, ...)
        The temporal dimension, where the leading axis means the batch size.
    key : JKey
        A JAX random key.

    Returns
    -------
    JArray, Callable[[JArray], dict], Callable (..., d), (p, ) -> (..., d)
        The initial parameter array, the array-to-dict ravel function, and the NN forward pass evaluation function.
        The dimension `d` is generic for any shape.
    """
    dict_param = neural_network.init(key, jnp.ones(shape_x), jnp.ones(shape_t))
    array_param, array_to_dict = ravel_pytree(dict_param)

    def forward_pass(x: JArray, t: FloatScalar, param: JArray) -> JArray:
        """The NN forward pass.
        x : (..., d)
        t : (...)
        param : (p, )
        return : (..., d)
        """
        return neural_network.apply(array_to_dict(param), x, t)

    return array_param, array_to_dict, forward_pass


def sinusoidal_embedding(k: JArray | FloatScalar, out_dim: int = 64, max_period: int = 10_000) -> JArray:
    """The so-called sinusoidal positional embedding.

    Parameters
    ----------
    k : FloatScalar
        A time variable. Note that this is in the discrete time.
    out_dim : int
        The output dimension.
    max_period : int
        The maximum period.

    Returns
    -------
    JArray (..., out_dim)
        An array.
    """
    half = out_dim // 2

    fs = jnp.exp(-math.log(max_period) * jnp.arange(half) / (half - 1))
    embs = k * fs
    embs = jnp.concatenate([jnp.sin(embs), jnp.cos(embs)], axis=-1)
    if out_dim % 2 == 1:
        raise NotImplementedError(f'out_dim is implemented for even number only, while {out_dim} is given.')
    return embs
