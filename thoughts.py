#!/usr/bin/env python3

__all__ = [
    'Thought',
    'new_thought_class',
]

import jax.numpy as jnp

from think.random import State


class BaseThought:

    STATE = State()

    THOUGHT_DIM = None

    def __init__(self, t=None):
        self._t = t

    def think(self):
        # initialize lazily
        if self._t is None:
            self._t = self._new_thought()
        return self._t

    def rethink(self, t):
        self._t = t

    def __repr__(self):
        return f"{self.__class__.__name__}({self._t})"

    def __array__(self):
        return self._t

    @classmethod
    def _new_thought(cls):
        denominator = jnp.sqrt(cls.THOUGHT_DIM)
        return cls.STATE.normal([cls.THOUGHT_DIM])/denominator

def new_thought_class(thought_dim, seed=42):

    class Thought(BaseThought):
        STATE = State(seed)
        THOUGHT_DIM = thought_dim

    return Thought


Thought = new_thought_class(1024)

