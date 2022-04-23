#!/usr/bin/env python3

__all__ = [
    'get_thought_dim',
    'set_thought_dim',
    'Thought',
]

import weakref
import jax.numpy as jnp
from .random import random_normal

THOUGHT_DIM = 1024

def get_thought_dim():
    return THOUGHT_DIM

def set_thought_dim(n):
    global THOUGHT_DIM
    THOUGHT_DIM = n
    return THOUGHT_DIM

def new_thought():
    return random_normal([THOUGHT_DIM])


class Thought:

    """
        A reference to a learnable variable.
        .think() gets the value
        .rethink() sets the value

        All learnable variables should use this
        extra layer of indirection, since jax
        arrays are immutible, and we need to be
        able to build types so we're not limited
        to tossing around tensors like cavemen.
    """

    def __new__(cls):
        self = object.__new__(cls)
        self._t = new_thought()
        cls.instances.add(self)
        return self

    def think(self):
        return self._t

    def rethink(self, t):
        self._t = t
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self._t})"

    def __array__(self):
        return self._t

    def __bool__(self):
        return True

    # for tracking memory usage
    instances = weakref.WeakSet()

    @classmethod
    def gigabytes(cls):
        bytes = 4*THOUGHT_DIM*len(cls.instances)
        return bytes/(1024**3)

    @classmethod
    def active(cls):
        return len(cls.instances)

