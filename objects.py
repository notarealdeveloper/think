#!/usr/bin/env python3

"""
    Objects you can .think() about.
"""

__all__ = [
    'Object',
    'Bool',
    'Str',
    'Int',
    'Float',
]


import abc
import numbers
import builtins

import slow
from think import Thought

OBJECTS = {}

class Object:

    type = object

    def __new__(cls, object, t=None):

        if not isinstance(object, cls.type):
            raise TypeError(f"object {object} is not of type {cls.type}")

        try:
            return OBJECTS[(cls, object)]
        except:
            # print(f"New object: {cls}, {object}")
            pass

        self = builtins.object.__new__(cls)
        OBJECTS[(cls, object)] = self
        return self

    def __init__(self, object, t=None):
        self.object  = object
        self.attrs   = {}
        self.thought = Thought(t)

    def think(self):
        return self.thought.think()

    def rethink(self, t):
        return self.thought.rethink(t)

    def set(self, attr, value, how='soft'):

        attr  = self._ensure_attr_is_type(attr)
        value = attr._ensure_value_is_object(value)

        if how == 'soft':
            self.attrs[attr] = value

        elif how == 'hard':
            t = slow.hardset(attr, self, value)
            self.rethink(t)

        elif how == 'both':
            self.hardset(attr, value)
            self.softset(attr, value)
        else:
            raise ValueError(f"how: {how!r}")

        return self

    def get(self, attr, how='soft'):

        attr = self._ensure_attr_is_type(attr)

        if attr not in self.attrs:
            # feeling
            thought = attr.project(self)
        else:
            # feeling and knowing
            value = self.attrs[attr]
            feel = attr.project(self)
            know = attr.project(value)
            thought = slow.mix([feel, know])

        if how == 'soft':
            return thought
        elif how == 'hard':
            return attr.invert(thought)
        else:
            raise ValueError(f"how: {how!r}")

    def _ensure_attr_is_type(self, attr):
        if not isinstance(attr, Type):
            raise TypeError(attr)
        return attr

    def _ensure_value_is_object(self, value):
        if not isinstance(value, Object):
            value = self(value)
        return value

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object})"

    def __array__(self):
        return self.think()

    def unwrap(self):
        while hasattr(self, 'object'):
            self = self.object
        return self


class Int(Object):
    type = int

class Float(Object):
    type = float

class Str(Object):
    type = str

class Bool(Object):
    type = bool


