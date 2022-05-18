#!/usr/bin/env python3

__all__ = [
    'Thought',
    'thought_dim',
    'random_normal',
    'new_thought',
    'new_thoughts',
    'ordinal_basis',
]

import jax.numpy as jnp

from think.internals import State

STATE = State()

THOUGHT_DIM = 1024

def random_normal(shape):
    return STATE.normal(shape)

def new_thought():
    denominator = jnp.sqrt(THOUGHT_DIM)
    return STATE.normal([THOUGHT_DIM])/denominator

def new_thoughts(n):
    denominator = jnp.sqrt(THOUGHT_DIM)
    return STATE.normal([n, THOUGHT_DIM])/denominator

def ordinal_basis(n, m=1):
    ortho = [new_thought() for k in range(n+1)]
    basis = []
    thetas = jnp.linspace(0, jnp.pi/2, m+1)[+1:-1]
    for a,c in zip(ortho[:-1], ortho[+1:]):
        bs = [a*jnp.cos(theta) + c*jnp.sin(theta) for theta in thetas]
        basis += [a,*bs]
    return jnp.stack(basis)

def thought_dim(n=None):
    global THOUGHT_DIM
    if n is not None:
        THOUGHT_DIM = n
    return THOUGHT_DIM

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
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}({self._t})"

    def __array__(self):
        return self.think()

