#!/usr/bin/env python3

__all__ = [
    'EnumType',
    'BoolType',
    'StrType',
    'IntType',
    'FloatType',
]

import slow
from think import Bool, Str, Int, Float, Type
import jax.numpy as jnp

class EnumType(Type):

    def __init__(cls, name):
        cls.memory = {}

    def __call__(cls, str, *args, **kwds):
        if str in cls.memory:
            return cls.memory[str]
        self = cls.object(str, *args, **kwds)
        cls.memory[str] = self
        # self.set(cls, self)
        return self

    def params(cls):
        return [v for v in cls.memory.values()]

    def _similarities(cls, object):
        vals    = [v.think() for k,v in cls.memory.items()]
        sims    = slow.pre_attention_l1(vals, object)
        return sims

    def similarities(cls, object):
        sims = cls._similarities(object)
        sims = [s.item() for s in sims]
        keys = [k for k,v in cls.memory.items()]
        pairs = list(zip(keys, sims))
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)

    def invert(cls, object):
        keys    = [k for k,v in cls.memory.items()]
        sims    = cls._similarities(object)
        idx     = int(jnp.argmax(sims))
        key     = list(keys)[idx]
        return key

    def project(cls, object):
        return slow.attention_l1(cls, object)

    def __array__(cls):
        vects = [slow.to_vector(o) for o in cls.params()]
        return jnp.stack(vects, axis=0)



class BoolType(EnumType):
    base = Bool

class StrType(EnumType):
    base = Str

class IntType(EnumType):
    base = Int

class FloatType(EnumType):
    base = Float

