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
import jax.numpy as jnp

import fast
import slow
from think import Thought
from think import Object
from think import Bool, Str, Int, Float

TYPES = {}

class Type(type, Object):

    type = type
    base = Object

    def __new__(cls, name, base=None, t=None):

        if base is None:
            base = cls.base

        if not isinstance(name, str):
            raise TypeError(f"Type's name must be a str")

        if not isinstance(base, type):
            raise TypeError(f"Type's base argument must be a class: {base}")

        if not issubclass(base, Object):
            raise TypeError(f"Type's base argument must subclass Object: {base}")

        try:
            return TYPES[(cls, name, base)]
        except:
            pass

        self = type.__new__(cls, name, (base,), {})
        self.name = name
        self.base = base
        self.object = self
        self.attrs   = {}
        self.thought = Thought(t)
        TYPES[(cls, name, base)] = self
        return self

    def __init__(self, name, base=Object, t=None):
        pass

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


