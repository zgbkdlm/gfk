"""
Simple MLPs.
"""
import jax
import flax.linen as nn
from jax import numpy as jnp
from gfk.nns.base import sinusoidal_embedding

nn_param_init = nn.initializers.xavier_normal()


class _CrescentTimeBlock(nn.Module):
    dt: float
    nfeatures: int

    @nn.compact
    def __call__(self, time_emb):
        time_emb = nn.Dense(features=self.nfeatures, kernel_init=nn_param_init)(time_emb)
        time_emb = nn.gelu(time_emb)
        time_emb = nn.Dense(features=self.nfeatures, kernel_init=nn_param_init)(time_emb)
        return time_emb


class CrescentMLP(nn.Module):
    """The MLP neural network construction used in the pedagogical example.
    """
    dt: float
    dim_out: int = 3
    hiddens = [64, 32, 16]

    @nn.compact
    def __call__(self, x, t):
        k = t / self.dt
        if t.ndim < 1:
            time_emb = jnp.expand_dims(sinusoidal_embedding(k, out_dim=64), 0)
        else:
            time_emb = jax.vmap(lambda z: sinusoidal_embedding(z, out_dim=64))(k)

        for h in self.hiddens:
            x = nn.Dense(features=h, kernel_init=nn_param_init)(x)
            x = (x * _CrescentTimeBlock(dt=self.dt, nfeatures=h)(time_emb) +
                 _CrescentTimeBlock(dt=self.dt, nfeatures=h)(time_emb))
            x = nn.gelu(x)
        x = nn.Dense(features=self.dim_out, kernel_init=nn_param_init)(x)
        return jnp.squeeze(x)
