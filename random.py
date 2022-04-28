#!/usr/bin/env python3

import jax.random

class State:
    def __init__(self, key=42):
        self.key = jax.random.PRNGKey(key)
    def split(self, key):
        key, subkey = jax.random.split(key)
        return key, subkey
    def normal(self, shape):
        self.key, subkey = self.split(self.key)
        return jax.random.normal(subkey, shape)

