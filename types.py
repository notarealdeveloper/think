#!/usr/bin/env python3

__all__ = [
    'EnumType',
]

import slow
from think import Object, Type
import jax.numpy as jnp

class EnumType(Type):

    def __init__(cls, name, bases=(), dict=None, **kwds):
        super().__init__(name, bases, dict, **kwds)
        cls.memory = {}

    def __call__(cls, obj):
        if obj in cls.memory:
            return cls.memory[obj]
        self = super().__call__(obj)
        cls.memory[obj] = self
        return self

    def params(cls):
        return [v for v in cls.memory.values()]

    def similarities(cls, object):
        pairs = cls.memory.items()
        keys = [k for k,v in pairs]
        vals = [v.think() for k,v in pairs]
        sims = slow.pre_attention_l1(vals, object)
        sims = [s.item() for s in sims]
        pairs = list(zip(keys, sims))
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)

    def invert(cls, object):
        keys = [k for k,v in cls.memory.items()]
        vals = [v.think() for k,v in cls.memory.items()]
        sims = slow.pre_attention_l1(vals, object)
        idx  = int(jnp.argmax(sims))
        key  = list(keys)[idx]
        return key

    def project(cls, object):
        return slow.attention_l1(cls, object)

    def __array__(cls):
        vects = [slow.to_vector(o) for o in cls.params()]
        return jnp.stack(vects, axis=0)

    def instances(cls):
        memory = {}
        for base in cls.subs:
            memory |= base.memory
        return memory

    def most_similar(cls, object):
        if not isinstance(object, cls):
            object = cls(object)
        others = cls.memory.values()
        sims = {}
        t_self = object.think()
        for other in others:
            if object == other:
                continue
            t_other = other.think()
            sims[other.object] = fast.cos(t_self, t_other).item()
        pairs = sorted(sims.items(), key=lambda pair: pair[1], reverse=True)
        return pairs

