#!/usr/bin/env python3

"""
    All2Vec core.

    The python type system.

    This time it's differentiable.
"""

# core objects
__all__ = [
    'Object',
    'Bool',
    'Str',
    'Int',
    'Float',
    'Subclass',
]

# core types
__all__ += [
    'Type',
    'BoolType',
    'StrType',
    'IntType',
    'FloatType',
]

import abc
import types
import logging
import numbers
import builtins

import fast
import slow
from think import Thought, new_thought
from think.internals import hybridmethod, metamethod
from think.ops import Add, Sub, Mul, Div

OBJECTS = {}
TYPES   = {}

logger = logging.getLogger(__name__)

S = '\n * '

class Type(type):

    """ A better way to implement Type. """

    object = type

    def __new__(meta, name, bases=(), dict=None, **kwds):
        logger.debug(f"{S}Type.__new__ (enter):"
                     f"{S}meta={meta}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     #f"{S}meta_dict={meta.__dict__}"
        )

        # dict
        if dict is None:
            dict = {}
        dict['__module__'] = None
        dict |= kwds

        # bases
        if not bases:
            if 'base' in dict: # compat
                bases = (dict['base'],)
            elif hasattr(meta, 'base'):
                bases = (meta.base,)
            elif hasattr(meta, 'firstborn'):
                bases = (meta.firstborn,)
            else:
                assert name == 'Object'
        elif isinstance(bases, type):
            bases = (bases,)
        else:
            for base in bases:
                assert isinstance(base, type)

        try:
            return TYPES[(meta, name, *bases)]
        except:
            pass

        # upgrade methods to metamethods
        for k,v in dict.items():
            if k.startswith('__') and k.endswith('__'):
                continue
            if isinstance(v, types.FunctionType):
                dict[k] = metamethod(v)
                logger.debug(f"{name}.{k} upgraded to metamethod")

        # create the class
        cls = super().__new__(meta, name, bases, dict)

        if meta is not Type:
            cls.object = meta.base.object

        if not hasattr(meta, 'firstborn'):
            meta.firstborn = cls

        logger.debug(f"{S}Type.__new__ (exit):"
                     f"{S}meta={meta}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     #f"{S}meta_dict={meta.__dict__}"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        TYPES[(meta, name, *bases)] = cls
        return cls

    def __init__(cls, name, bases=(), dict=None, **kwds):
        logger.debug(f"{S}Type.__init__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        super().__init__(name, bases, dict)
        cls.name = name
        cls.bases = [base for base in cls.mro() if hasattr(base, 'think')]
        cls.attrs = {}
        cls.thought = Thought()
        cls.kwds = kwds
        cls.subs = []
        logger.debug(f"{S}Type.__init__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        return None

    def __not_really_call__(cls, *args, **kwds):
        logger.debug(f"{S}Type.__call__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     #f"{S}cls_dict={cls.__dict__}"
        )

        self = super().__call__(*args, **kwds)

        logger.debug(f"{S}Type.__call__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     #f"{S}cls_dict={cls.__dict__}"
                     f"{S}self={self}"
                     f"\n"
        )
        return self

    def parent_thoughts(cls):
        return [base.thought for base in cls.bases]

    def think(cls):
        t = cls.thought
        Ts = cls.parent_thoughts()
        if not Ts:
            return t.think()
        else:
            T = slow.mix(Ts)
            return slow.mix([T, t])

    def rethink(cls, t):
        cls.thought.rethink(t)
        return cls


class Object(metaclass=Type):

    object = object

    """ An Object you can .think() about. """

    def __new__(cls, object=None, t=None):

        if not isinstance(object, cls.object):
            raise TypeError(f"object {object} is not of type {cls.object}")

        try:
            return OBJECTS[(cls, object)]
        except:
            # print(f"New object: {cls}, {object}")
            pass

        self = builtins.object.__new__(cls)
        OBJECTS[(cls, object)] = self
        self.object  = object
        self.attrs   = {}
        self.thought = Thought()
        self.type    = type(cls)
        return self

    def __init__(self, object=None, t=None):
        pass

    def __init_subclass__(cls, **kwds):
        logger.debug(f"{S}Object.__init_subclass__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}kwds={kwds}"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        super().__init_subclass__()
        for base in cls.bases:
            base.subs.append(cls)
        logger.debug(f"{S}Object.__init_subclass__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     #f"{S}cls_dict={cls.__dict__}"
        )

    @metamethod
    def think(self):
        T = type(self).think()
        t = self.thought.think()
        return slow.mix([T, t])

    @metamethod
    def rethink(self, t):
        self.thought.rethink(t)
        return self

    def setfeel(self, attr, value):
        value = self._ensure_value_is_attr_instance(attr, value)
        t = slow.setattr(attr, self, value)
        self.rethink(t)
        return self

    def setknow(self, attr, value):
        value = self._ensure_value_is_attr_instance(attr, value)
        self.attrs[attr] = value
        return self

    def set(self, attr, value):
        attr  = self._ensure_attr_is_object_subclass(attr)
        value = self._ensure_value_is_attr_instance(attr, value)
        self.setfeel(attr, value)
        self.setknow(attr, value)
        return self

    def getfeel(self, attr, hard=False):
        thought = attr.project(self)
        return thought if not hard else attr.invert(thought)

    def getknow(self, attr, hard=False):
        thought = self.attrs.get(attr, Object(None))
        return thought if not hard else attr.invert(thought)

    def getboth(self, attr, hard=False):
        feel = self.getfeel(attr)
        know = self.getknow(attr)
        if not know:
            thought = feel
        else:
            thought = slow.mix([feel, know])
        return thought if not hard else attr.invert(thought)

    def get(self, attr, how='feel', hard=True):
        attr = self._ensure_attr_is_object_subclass(attr)
        if how == 'feel':
            return self.getfeel(attr, hard=hard)
        elif how == 'know':
            return self.getknow(attr, hard=hard)
        elif how == 'both':
            return self.getboth(attr, hard=hard)
        else:
            raise ValueError(f"how: {how!r}")

        if hard:
            return attr.invert(thought)
        else:
            return thought

    def __array__(self):
        return self.think()

    def __eq__(self, other):
        return type(self) == type(other) and self.object == other.object

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object})"

    def __bool__(self):
        return self.object is not None

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Mul(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def _ensure_attr_is_object_subclass(self, attr):
        if not is_subclass_of_object(attr):
            raise TypeError(attr)
        return attr

    def _ensure_value_is_attr_instance(self, attr, value):
        if not is_instance_of_object(value):
            value = attr(value)
        elif value.type is not attr:
            # experimental, e.g., Dirname(Pathname('/etc/security'))
            # value = attr(value.unwrap())
            value = value.unwrap()
        return value

    def unwrap(self):
        while hasattr(self, 'object'):
            self = self.object
        return self

    @classmethod
    def ensure_unwrapped(cls, object):
        return cls.unwrap(object)


def is_instance_of_object(o):
    return isinstance(o, Object)

def is_subclass_of_object(o):
    return is_instance_of_object(o) and (o.type is Type)

def Subclass(*bases, name='Unnamed'):
    return Type(name, bases)


# types.py


# the solution to infinite regress is self reference.
# 1. type's type is type
# 2. type's base is object
# 3. object's base is object
# 4. object's type is type
# https://youtu.be/uOzdG3lwcB4?t=3209
# (see the 30 seconds from 53:30 to 54:00)
Type.type = Type
Object.type = Type
Type.base = Object
Object.base = Object


class Int(Object):
    object = int

class Float(Object):
    object = float

class Str(Object):
    object = str

class Bool(Object):
    object = bool

class BoolType(Type):
    base = Bool

class StrType(Type):
    base = Str

class IntType(Type):
    base = Int

class FloatType(Type):
    base = Float

