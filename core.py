#!/usr/bin/env python3

# core.py

"""
    All2Vec core.

    The python type system.

    This time it's differentiable.
"""

__all__ = [
    'Object',
    'Type',
]

import abc
import types
import logging
import numbers
import builtins
import itertools
import jax.numpy as jnp

import fast
import slow
import think
from think import Thought, new_thought
from think.internals import hybridmethod, metamethod
from think.ops import Add, Sub, Mul, Div

OBJECTS = {}
TYPES   = {}

# all 4 combinations of these two boolean variables
# should produce identical outcomes if the system is
# consistent.
DEFINE_OBJECTS_USING_CLASSES = True
DEFINE_TYPES_USING_CLASSES = True

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
        dict = {} if dict is None else dict
        dict |= kwds

        # bases
        if not bases:
            if 'base' in dict:
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

        #try:
        #    return TYPES[(meta, name, *bases)]
        #except:
        #    pass

        # upgrade methods to metamethods
        for k,v in dict.items():
            if k.startswith('__') and k.endswith('__'):
                continue
            if isinstance(v, types.FunctionType):
                dict[k] = metamethod(v)
                logger.debug(f"{name}.{k} upgraded to metamethod")

        # create the class
        cls = super().__new__(meta, name, bases, dict)
        cls.__module__ = None

        if not hasattr(meta, 'firstborn'):
            meta.firstborn = cls

        logger.debug(f"{S}Type.__new__ (exit):"
                     f"{S}meta={meta}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     f"{S}meta_dict={meta.__dict__}"
                     f"{S}cls_dict={cls.__dict__}"
        )
        cls.name = name
        cls.bases = [base for base in cls.mro() if hasattr(base, 'think')]
        cls.attrs = {}
        cls.thought = Thought()
        cls.kwds = kwds
        cls.subs = []
        #TYPES[(meta, name, *bases)] = cls
        return cls

    def __init__(cls, name, bases=(), dict=None, **kwds):
        logger.debug(f"{S}Type.__init__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
        )
        super().__init__(name, bases, dict)
        # Experimental: all types are memory types
        cls.memory = {}
        logger.debug(f"{S}Type.__init__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"\n"
                     f"{S}cls_dict={cls.__dict__}"
        )
        return None

    def __call__(cls, obj, *args, **kwds):
        logger.debug(f"{S}Type.__call__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
        )

        if obj in cls.memory:
            return cls.memory[obj]
        # self = super().__call__(obj, *args, **kwds)
        self = cls.__new__(cls, obj, *args, **kwds)
        if isinstance(self, cls):
            cls.__init__(self, obj, *args, **kwds)
        cls.memory[obj] = self

        logger.debug(f"{S}Type.__call__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
                     f"{S}self={self}"
                     f"\n"
        )
        return self

    @metamethod
    def mro(cls):
        return type.mro(cls)

    __module__ = None
    def __init_subclass__(meta):
        meta.__module__ = None

    def think(cls):
        t = cls.thought
        Ts = [base.thought for base in cls.bases]
        if not Ts:
            return t.think()
        else:
            T = slow.mix(Ts)
            return slow.mix([T, t])

    def rethink(cls, t):
        cls.thought.rethink(t)
        return cls

    def name(cls, name=None):
        if name is None:
            return cls.__qualname__
        else:
            cls.__qualname__ = name
        return cls

    def __dir__(cls):
        # dir normally contains keys from:
        # * self.__dict__
        # * cls.__dict__ for cls in self.__class__.mro()
        # for a class produced by a metaclass, this should be:
        self_dir  = [list(cls.__dict__)]
        base_dirs = [list(base.__dict__) for base in cls.mro()]
        all_dirs  = self_dir + base_dirs
        dir = sum(all_dirs, [])
        dir = sorted(set(dir))
        return dir

    # begin experimental: for allowing all types to be memory types by default

    def instances(cls):
        instances = {}
        for base in cls.subs:
            instances |= base.memory
        return instances

    def similarities(cls, object):
        print('similarities called')
        pairs = cls.memory.items()
        keys = [k for k,v in pairs]
        vals = [v.think() for k,v in pairs]
        sims = slow.pre_attention_l1(vals, object)
        sims = [s.item() for s in sims]
        pairs = list(zip(keys, sims))
        return sorted(pairs, key=lambda pair: pair[1], reverse=True)

    def invert(cls, object):
        pairs = cls.memory.items()
        keys = [k for k,v in pairs]
        vals = [v.think() for k,v in pairs]
        sims = slow.pre_attention_l1(vals, object)
        idx  = int(jnp.argmax(sims))
        key  = list(keys)[idx]
        return key

    def project(cls, object):
        return slow.attention_l1(cls, object)

    def __array__(cls):
        vects = [slow.to_vector(o) for o in cls.memory.values()]
        return jnp.stack(vects, axis=0)

    # begin experimental: for allowing all types to be memory types by default


class Object(metaclass=Type):

    object = object

    """ An Object you can .think() about. """

    def __new__(cls, object=None, t=None):

        if isinstance(object, Type):
            name = f"{cls.__name__}({object.__name__})"
            bases = (object,)
            return type(cls)(name, bases, {})

        if not isinstance(object, cls.object):
            raise TypeError(f"object {object} is not of type {cls.object}")

        try:
            return OBJECTS[(cls, object)]
        except:
            pass

        self = builtins.object.__new__(cls)
        OBJECTS[(cls, object)] = self
        self.object  = object
        self.attrs   = {}
        self.thought = Thought()
        self.type    = type(cls)
        return self

    def __init__(self, *args, **kwds):
        pass

    def __init_subclass__(cls, **kwds):
        logger.debug(f"{S}Object.__init_subclass__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
        )
        super().__init_subclass__()
        for base in cls.bases:
            base.subs.append(cls)
        logger.debug(f"{S}Object.__init_subclass__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
                     f"\n"
        )

    def think(self):
        return self.thought.think()

    def rethink(self, t):
        self.thought.rethink(t)
        return self

    def setfeel(self, attr, value):
        value = attr._ensure_value_is_object(value)
        t = slow.setattr(attr, self, value)
        self.rethink(t)
        return self

    def setknow(self, attr, value):
        value = attr._ensure_value_is_object(value)
        self.attrs[attr] = value
        return self

    def set(self, attr, value):
        value = attr._ensure_value_is_object(value)
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

    @metamethod
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

    @classmethod
    def _ensure_value_is_object(cls, value):
        if not isinstance(value, Object):
            value = cls(value)
        return value

    def unwrap(self):
        while hasattr(self, 'object'):
            self = self.object
        return self

    @classmethod
    def ensure_unwrapped(cls, object):
        return cls.unwrap(object)

    def __class_getitem__(cls, n):
        """ This is how to handle sequences. """
        try:
            return cls.subclasses[n]
        except AttributeError:
            cls.subclasses = {}
        except KeyError:
            pass
        name = f"{cls.__name__}{n}"
        sub = Type(name, cls)
        cls.subclasses[n] = sub
        return sub


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


# objects.py

# from think.core import Object, Type

__all__ += [
    'Bool',
    'Str',
    'Int',
    'Float',
    'Bytes',
    'Complex',
]

if DEFINE_OBJECTS_USING_CLASSES:
    class Str(Object):
        object = str

    class Int(Object):
        object = int

    class Bool(Int):
        # bool.mro() is [bool, int, object]
        object = bool

    class Bytes(Object):
        object = bytes

    class Float(Object):
        object = float

    class Complex(Object):
        object = complex


else:
    Str     = Type('Str',     Object, object=str)
    Int     = Type('Int',     Object, object=int)
    Bool    = Type('Bool',    Int,    object=bool) # bool.mro() is [bool, int, object]
    Bytes   = Type('Bytes',   Object, object=bytes)
    Float   = Type('Float',   Object, object=float)
    Complex = Type('Complex', Object, object=complex)


# types.py

# from think.core import Object, Type
# from think.objects import Bool, Str, Int, Float, Bytes, Complex

__all__ += [
    'BoolType',
    'StrType',
    'IntType',
    'FloatType',
    'BytesType',
    'ComplexType',
]

if DEFINE_TYPES_USING_CLASSES:
    class BoolType(Type):
        base = Bool

    class StrType(Type):
        base = Str

    class IntType(Type):
        base = Int

    class FloatType(Type):
        base = Float

    class BytesType(Type):
        base = Bytes

    class ComplexType(Type):
        base = Complex

else:
    BoolType    = type('BoolType', (Type,), {'base':Bool})
    StrType     = type('StrType', (Type,), {'base':Str})
    IntType     = type('IntType', (Type,), {'base':Int})
    FloatType   = type('FloatType', (Type,), {'base':Float})
    BytesType   = type('BytesType', (Type,), {'base':Bytes})
    ComplexType = type('ComplexType', (Type,), {'base':Complex})


# resolve the infinite loop by overwriting the metaclass pointer
#Str.__class__       = StrType
#Int.__class__       = IntType
#Float.__class__     = FloatType
#Bytes.__class__     = BytesType
#Bool.__class__      = BoolType
#Complex.__class__   = ComplexType

# thus ends the core
# all else is commentary

