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
    'IsInstance',
]

import abc
import types
import logging
import numbers
import logging
import builtins
import itertools
import jax.numpy as jnp

import fast
import slow
import think
from think import Thought, new_thought
from think.internals import hybridmethod, metamethod
from think.ops import Add, Sub, Mul, Div
from think import gradients
from think.pretty import colors


OBJECTS = {}
TYPES   = {}

# all 4 combinations of these two boolean variables
# should produce identical outcomes if the system is
# consistent.
DEFINE_OBJECTS_USING_CLASSES = True
DEFINE_TYPES_USING_CLASSES = True

# if this is set to True, then all types allow None as a value
ALL_TYPES_NULLABLE = True

logger = logging.getLogger(__name__)


### since we need to store self.object in cls.memory,
### we need hashable subclasses of unhashable builtin
### python types, and the user shouldn't have to know
### about these things.
###
### note: the hashes of python scalars are not constant
### from one python process to another, so we need a
### stronger form of serialization for the saving and
### loading step. worst case scenario, we can just
### pickle the damn things here, and recompute when
### they change.
class thinklist(list):
    def __hash__(self):
        return hash(tuple(self))

class thinkdict(dict):
    def __hash__(self):
        return hash(tuple(self.items()))

class thinkset(set):
    def __hash__(self):
        return hash(tuple(self))

TYPE_PROMOTIONS = {
    list: thinklist,
    set: thinkset,
    dict: thinkdict,
}

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
        cls.contexts = {}
        cls.memory = {}

        # list, set, dict:
        if cls.object in TYPE_PROMOTIONS and not hasattr(cls, '__object__'):
            cls.__object__ = TYPE_PROMOTIONS[cls.object]

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

        if hasattr(cls, '__instances__'):
            assert isinstance(cls.__instances__, (tuple, list, set))
            for o in cls.__instances__:
                assert not isinstance(o, Object)
                cls(o)
            del cls.__instances__

        logger.debug(f"{S}Type.__init__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}name={name}"
                     f"{S}bases={bases}"
                     f"{S}dict={dict}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
                     f"\n"
        )

        return None

    def __call__(cls, arg, *args, **kwds):
        logger.debug(f"{S}Type.__call__ (enter):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
        )

        # __object__
        #
        # the think-specific magic method __object__, if it exists,
        # will be called before the arguments are passed to __new__.
        # in this sense it is analogous to the python magic method
        # __prepare__ in normal python metaclasses.
        if hasattr(cls, '__object__'):
            object = cls.__object__(arg, *args, **kwds)
        else:
            object = arg

        if object in cls.memory:
            return cls.memory[object]

        # make the instance
        self = cls.__new__(cls, object, *args, **kwds)

        # __raw__
        #
        # the think-specific magic attribute __raw__ stores the raw argument
        # that was passed to meta.__call__. This is useful in cases when the
        # user defines __object__ to make a class accept polymorphic inputs,
        # but still wants access to the raw argument that was passed.
        # For example, if we have:
        #
        # class Sentence(List[Word]):
        #
        #     ...
        #
        #     def __repr__(self):
        #         return f"Sentence({self.__raw__})"
        #
        # which we want to be able to instantiate by passing a string, not just a list
        # of words, then that string will be stored in __raw__.
        #
        self.__raw__ = arg
        if isinstance(self, cls):
            cls.__init__(self, object, *args, **kwds)

        # use the key self.object in case the class author decided to set
        # the .object attribute to something other than what was passed in.
        # this is a fairly common occurrence, so this step is important
        cls.memory[object] = self

        logger.debug(f"{S}Type.__call__ (exit):"
                     f"{S}cls={cls}"
                     f"{S}args={args}"
                     f"{S}kwds={kwds}"
                     f"{S}cls_dict={cls.__dict__}"
                     f"{S}self={self}"
                     f"\n"
        )

    # <experimental Instance property>

        # experimental: resolving the tension between the class level thoughts idea
        # and the general theme of setattr by implementing "isinstance" using setattr.
        if cls.primary:
            for base in cls.bases:
                if base is Object: continue
                self.set(IsInstance[base], True)
        return self

    # </experimental Instance property>

    def mro(cls):
        return type.mro(cls)

    def __repr__(cls):
        return f"{cls.__qualname__}"

    __module__ = None
    def __init_subclass__(meta):
        meta.__module__ = None

    def think(cls):
        return cls.thought.think()

    def rethink(cls, t):
        cls.thought.rethink(t)
        return cls

    def __dir__(cls):
        # for normal object, dir contains keys from:
        # * self.__dict__
        # * cls.__dict__ for cls in self.__class__.mro()
        # so if we want classes to behave like normal objects w.r.t. tab completion,
        # the __dir__ for a class produced by a metaclass should be:
        self_dir  = [list(cls.__dict__)]
        base_dirs = [list(base.__dict__) for base in cls.mro()]
        all_dirs  = self_dir + base_dirs
        dir = sum(all_dirs, [])
        dir = sorted(set(dir))
        return dir

    def instances(cls):
        instances = {}
        instances |= cls.memory
        for base in cls.subs:
            instances |= base.memory
        return instances

    primary = True

    ######################################
    ### plumbing for getattr / setattr ###
    ######################################

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
        sims = slow.dots(vals, object) # quicker version of slow.pre_attention_l1(vals, object)
        idx  = int(jnp.argmax(sims))
        key  = list(keys)[idx]
        return key

    def project(cls, object):
        return slow.attention_l1(cls, object)

    def __array__(cls):
        if len(cls.memory) == 0:
            raise Exception(f"Attribute {cls!r} doesn't yet have any instances.")
        vects = [o.__array__() for o in cls.memory.values()]
        return jnp.stack(vects, axis=0)

    ####################################
    ### self training code for types ###
    ####################################
    def perfect(cls):
        for name, self in cls.instances().items():
            if self.score() < 1.0:
                return False
        return True

    def learn(cls, *args, **kwds):
        while not cls.perfect():
            for name, self in cls.instances().items():
                gradients.learn_until_score(self, threshold=1.0, *args, **kwds)
        if cls is not Object:
            print(colors.white(f"{cls.name} is now perfect ✨"))
        else:
            print(colors.white(f"The system is now perfect ✨"))

    def learn_until_loss(cls, *args, **kwds):
        for name, self in cls.instances().items():
            gradients.learn_until_loss(self, *args, **kwds)

    def learn_until_score(cls, *args, **kwds):
        for name, self in cls.instances().items():
            gradients.learn_until_score(self, *args, **kwds)


class Object(metaclass=Type):

    object = object

    """ An Object you can .think() about. """

    def __new__(cls, object=None, t=None):

        if isinstance(object, Type):
            name = f"{cls.__name__}({object.__name__})"
            bases = (object,)
            return type(cls)(name, bases, {})

        if ALL_TYPES_NULLABLE:
            if not isinstance(object, cls.object) and object is not None:
                raise TypeError(f"object {object} is not of type {cls.object}")
        else:
            if not isinstance(object, cls.object):
                raise TypeError(f"object {object} is not of type {cls.object}")

        self = builtins.object.__new__(cls)
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
        value = attr.ensure_thinkable(value)
        t = slow.setattr(attr, self, value)
        self.rethink(t)
        return self

    def setknow(self, attr, value):
        value = attr.ensure_thinkable(value)
        self.attrs[attr] = value
        return self

    def set(self, attr, value):
        value = attr.ensure_thinkable(value)
        # without the setfeel line, gradients are responsible for everything,
        # and the system is MUCH less able to learn quickly and sometimes
        # doesn't even converge.
        self.setfeel(attr, value)
        self.setknow(attr, value)
        return self

    def getfeel(self, attr, hard=False):
        thought = attr.project(self)
        return thought if not hard else attr.invert(thought)

    def getknow(self, attr, hard=False):
        thought = self.attrs.get(attr, Object(None))
        return thought if not hard else attr.invert(thought)

    def get(self, attr, how='feel', hard=True):
        if how == 'feel':
            return self.getfeel(attr, hard=hard)
        elif how == 'know':
            return self.getknow(attr, hard=hard)
        else:
            raise ValueError(f"how: {how!r}")

    def Get(self, attr):
        return attr(self.getfeel(attr, hard=True))

    @metamethod
    def __array__(self):
        return self.think()

    def __eq__(self, other):
        if not isinstance(other, Object):
            return type(self) == type(other) and self.object == other.object
        else:
            return type(self.object) == type(other) and self.object == other

    def __repr__(self):
        return colors.blue(f"{self.object}")

    def __str__(self):
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

    def unwrap(self):
        while hasattr(self, 'object'):
            self = self.object
        return self

    @classmethod
    def ensure_thinkable(cls, value):
        if isinstance(value, Object):
            value = value.unwrap()
        self = cls(value)

        for sup in cls.bases:
            if sup.primary:
                sup(value)
                break # good, this works, that solves the mystery
        return self

    @classmethod
    def ensure_unthinkable(cls, object):
        return cls.unwrap(object)

    @classmethod
    def __class_getitem__(cls, item):
        """ This is how to handle sequences. """
        # these are contextual types
        try:
            return cls.contexts[item]
        except KeyError:
            pass
        name = f"{cls.__name__}[{item}]"
        if isinstance(item, Type):
            # List[Str] is a subclass of List whose .Item is set to Str
            sub = Type(name, cls, primary=False, Item=item)
        elif isinstance(item, tuple) and len(item) == 2 \
        and isinstance(item[0], Type) \
        and isinstance(item[1], Type):
            # Dict[Str, Int] is a subclass of Dict whose .Key is Str and .Value is Int
            key, value = item
            name = f"{cls.__name__}[{key}, {value}]]"
            sub = Type(name, cls, primary=False, Key=key, Value=value)
        else:
            # List[3] is a subclass of List whose .item is set to 3
            sub = Type(name, cls, primary=False, context=item)
        cls.contexts[item] = sub
        return sub

    def reset_wrong(self):
        wrong = []
        for attr, value in self.attrs.items():
            feel = self.get(attr)
            know = value.object
            if feel != know:
                wrong.append([attr, value])
        for attr, value in wrong:
            self.setfeel(attr, value)

    def score(self, reset_wrong=False):
        if not self.attrs:
            return 1.0
        right = []
        wrong = []
        for attr, value in self.attrs.items():
            feel = self.get(attr)
            know = value.object
            if feel == know:
                right.append([attr, value])
            else:
                wrong.append([attr, value])

        # compute score before resetting anything, so it's pure
        score = len(right)/len(self.attrs)

        # now reset the wrong ones if asked
        if reset_wrong:
            for attr, value in wrong:
                self.setfeel(attr, value)
        return score


    learn_until_loss  = gradients.learn_until_loss
    learn_until_score = gradients.learn_until_score

    @metamethod
    def perfect(self):
        return self.score() == 1.0

    @metamethod
    def learn(self):
        gradients.learn_until_score(self, threshold=1.0)
        print(colors.white(f"{self} is now perfect ✨"))


# In python:
#
# 1. type's type is type
# 2. type's base is object
# 3. object's base is object
# 4. object's type is type
#
# (see 53:30 to 54:00 here: https://youtu.be/uOzdG3lwcB4?t=3209)
#
# tl;dr: the solution to infinite regress is self reference.
Type.type = Type
Object.type = Type
Type.base = Object
Object.base = Object


# objects.py

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
    Bool    = Type('Bool',    Int,    object=bool)
    Bytes   = Type('Bytes',   Object, object=bytes)
    Float   = Type('Float',   Object, object=float)
    Complex = Type('Complex', Object, object=complex)


class IsInstance(Bool):
    ...


# types.py

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



__all__ += [
    'perfect',
    'learn',
]

perfect = Object.perfect
learn   = Object.learn

# thus ends the core
# all else is commentary

