#!/usr/bin/env python3

"""
    All2Vec core.

    The python type system.

    This time it's differentiable.
"""

# core objects
__all__ = [
    'T',
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

class T(type):

    """ A better way to implement Type. """

    firstborn = None

    def __new__(meta, name='', bases=(), dict=None, **kwds):
        logger.debug(f"{S}T.__new__ (enter):"
                     f"{S}meta={meta}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     #f"{S}meta_dict={meta.__dict__}"
        )
        if not bases and meta.firstborn is not None:
            bases = (meta.firstborn,)
        if dict is None:
            dict = {}
        dict['__module__'] = None
        cls = super().__new__(meta, name, bases, dict, **kwds)
        for k,v in dict.items():
            if k.startswith('__') and k.endswith('__'):
                continue
            if isinstance(v, types.FunctionType):
                setattr(cls, k, metamethod(v))
                logger.debug(f"{cls.__name__}.{k} upgraded to metamethod")
        if meta.firstborn is None:
            meta.firstborn = cls
        logger.debug(f"{S}T.__new__ (exit):"
                     f"{S}meta={meta}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     #f"{S}meta_dict={meta.__dict__}"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        return cls

    def __init__(cls, name='', bases=(), dict=None, **kwds):
        logger.debug(f"{S}T.__init__ (enter):"
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
        logger.debug(f"{S}T.__init__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        return None

    def __call__(cls, *args, **kwds):
        logger.debug(f"{S}T.__call__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     #f"{S}cls_dict={cls.__dict__}"
        )

        self = super().__call__(*args, **kwds)

        logger.debug(f"{S}T.__call__ (exit):"
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


class Object(metaclass=T):

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
        self.type    = getattr(cls, 'meta', cls) # for derived objects
        return self

    def __init__(self, object=None, t=None):
        pass

    def __init_subclass__(cls, **kwds):
        logger.debug(f"{S}O.__init_subclass__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}kwds={kwds}"
                     #f"{S}cls_dict={cls.__dict__}"
        )
        super().__init_subclass__()
        for base in cls.bases:
            base.subs.append(cls)
        logger.debug(f"{S}O.__init_subclass__ (exit):"
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object})"

    def __bool__(self):
        return self.object is not None

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

def Subclass(*bases, name=''):
    return T(name, bases)


# types.py


class Type(Object):

    object = type

    def __new__(cls, name, base=None, t=None):

        if not base:
            base = getattr(cls, 'base', Object)

        if isinstance(base, tuple):
            try:    [base] = base
            except: raise NotImplementedError(f"No multiple inheritance yet.")

        if not isinstance(name, str):
            raise TypeError(f"Type's name must be a str")

        if not isinstance(base, type):
            raise TypeError(f"Type's base argument must be a class: {base}")

        if not issubclass(base, Object):
            raise TypeError(f"Type's base argument must subclass Object: {base}")

        if self := TYPES.get((cls, name, base)):
            return self

        # initialize the core attributes here
        # so the user doesn't have to remember
        # to call super().__init__ or anything.
        # This way, everything should Just Work.
        self = object.__new__(cls)
        self.name    = name
        self.base    = base

        self.object  = type(name, (base,), {'meta': self})
        self.attrs   = {}
        self.thought = Thought(t)

        TYPES[(cls, name, base)] = self
        return self

    def __init__(self, name, base=None, t=None):
        pass

    def __call__(self, object):
        return self.object(object)

    def __add__(self, object):
        return Add(self, object)

    def __sub__(self, object):
        return Sub(self, object)

    def __mul__(self, object):
        return Mul(self, object)

    def __truediv__(self, object):
        return Div(self, object)

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

