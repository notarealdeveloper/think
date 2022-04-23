#!/usr/bin/env python3

import random
import jax.random

def gen_jax_prng_keys(seed=None):
    seed = random.randint(0, 2**64)
    key = jax.random.PRNGKey(seed)
    while True:
        yield key
        key, subkey = jax.random.split(key)

def random_normal(shape):
    key = next(JAX_PRNG_KEYS)
    return jax.random.normal(key, shape)

JAX_PRNG_KEYS = gen_jax_prng_keys()
