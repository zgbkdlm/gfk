import jax
import numpy as np

master_key = jax.random.PRNGKey(666)

keys = jax.random.split(master_key, num=10000)
np.save('rnd_keys', keys)
