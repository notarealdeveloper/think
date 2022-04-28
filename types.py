#!/usr/bin/env python3

# types.py

__all__ = [
    'Type',
    'BoolType',
    'StrType',
    'IntType',
    'FloatType',
]

import abc
import types
import builtins
import jax.numpy as jnp

import fast
import slow
from think import Thought
from think import Object
from think import Bool, Str, Int, Float

TYPES = {}

def meta(obj):
    # me need a way of checking who values came from
    # without the neverending headache of dealing with
    # actual metaclasses, which aren't quite there.
    try:
        return obj.meta
    except:
        return type(obj)

class Type(Object):

    type = type
    base = Object

    def __new__(cls, name, base=None, t=None):

        if not base:
            base = cls.base

        if isinstance(base, tuple) and base:
            assert len(base) == 1
            [base] = base

        if not isinstance(name, str):
            raise TypeError(f"Type's name must be a str")

        if not isinstance(base, builtins.type):
            breakpoint()
            raise TypeError(f"Type's base argument must be a class: {base}")

        if not issubclass(base, Object):
            raise TypeError(f"Type's base argument must subclass Object: {base}")

        try:
            return TYPES[(cls, name, base)]
        except:
            pass

        object = type(name, (base,), {})
        self = builtins.object.__new__(cls) #Object.__new__(cls, object)
        self.name = name
        self.base = base
        self.object = object
        self.attrs   = {}
        self.thought = Thought(t)
        TYPES[(cls, name, base)] = self
        return self

    def __init__(self, name, base=None, t=None):
        pass

    def __call__(self, object):
        obj = self.object(object)
        obj.meta = self # this isn't general enough.
        return obj

    def __repr__(self):
        return f"{self.name}"

    @abc.abstractmethod
    def project(self, object):
        raise NotImplementedError

    @abc.abstractmethod
    def invert(self, object):
        raise NotImplementedError

    @abc.abstractmethod
    def params(self):
        raise NotImplementedError


class BoolType(Type):
    base = Bool

class StrType(Type):
    base = Str

class IntType(Type):
    base = Int

class FloatType(Type):
    base = Float


