#!/usr/bin/env python3

__all__ = [
    'Thought',
    'set_thought_dim',
]

import jax.numpy as jnp

from think.random import State

STATE = State()

THOUGHT_DIM = 1024

def new_thought():
    denominator = jnp.sqrt(THOUGHT_DIM)
    return STATE.normal([THOUGHT_DIM])/denominator

def set_thought_dim(n):
    global THOUGHT_DIM
    THOUGHT_DIM = n

class Thought:

    def __init__(self, t=None):
        self._t = t

    def think(self):
        # initialize lazily
        if self._t is None:
            self._t = new_thought()
        return self._t

    def rethink(self, t):
        self._t = t

    def __repr__(self):
        return f"{self.__class__.__name__}({self._t})"

    def __array__(self):
        return self._t

