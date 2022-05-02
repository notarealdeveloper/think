#!/usr/bin/env python3

__all__ = [
    'hybridmethod',
]

# random

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


# descriptors

class hybridmethod:

    def __init__(self, func):
        self.func = func

    def __get__(self, object, type):
        if object is None:
            return self.func.__get__(type)
        else:
            return self.func.__get__(object)


