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
    'save_system',
    'load_system',
]

import types
import logging
import builtins
import itertools
import jax.numpy as jnp

from think import slow
from think import Thought
from think.internals import metamethod
from think.ops import Add, Sub, Mul, Div
from think import gradients
from think.pretty import colors

# all 4 combinations of these two boolean variables
# should produce identical outcomes if the system is
# consistent.
DEFINE_OBJECTS_USING_CLASSES = True
DEFINE_TYPES_USING_CLASSES = True

# if this is set to True, then all types allow None as a value
ALL_TYPES_NULLABLE = True

logger = logging.getLogger(__name__)


# hashability
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

def sort_dict(d):
    return dict(sorted(d.items(), key=lambda pair: pair[1], reverse=True))


class Type(type):

    """ A better way to implement Type. """

    object = type

    def __new__(meta, name, bases=(), dict=None, **kwds):

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

        # upgrade methods to metamethods
        for k,v in dict.items():
            if k.startswith('__') and k.endswith('__'):
                continue
            if isinstance(v, types.FunctionType):
                dict[k] = metamethod(v)
                logger.debug(f"{name}.{k} upgraded to metamethod")

        # create the class
        cls = super().__new__(meta, name, bases, dict)

        if not hasattr(meta, 'firstborn'):
            meta.firstborn = cls

        cls.name = name
        cls.bases = [base for base in cls.mro() if hasattr(base, 'think')]
        cls.attrs = {}
        cls.thought = Thought()
        cls.kwds = kwds
        cls.subs = []
        cls.contexts = {}

        # list, set, dict:
        if cls.object in TYPE_PROMOTIONS and not hasattr(cls, '__object__'):
            cls.__object__ = TYPE_PROMOTIONS[cls.object]

        return cls

    def __init__(cls, name, bases=(), dict=None, **kwds):

        super().__init__(name, bases, dict)

        cls.memory = {}

        if hasattr(cls, '__create__'):
            assert isinstance(cls.__create__, (tuple, list, set))
            for o in cls.__create__:
                assert not isinstance(o, Object)
                cls(o)

        return None

    def __call__(cls, arg, *args, **kwds):

        object = cls._coerce_object(arg, *args, **kwds)

        if cls._is_hashable(object):
            cached = cls._lookup_cached(object)
            if cached is not None:
                return cached

        self = cls.__new__(cls, object, *args, **kwds)
        self.__raw__ = arg

        if isinstance(self, cls):
            cls.__init__(self, object, *args, **kwds)

        cls._register_instance(object, self)
        return self

    def _coerce_object(cls, arg, *args, **kwds):
        if hasattr(cls, '__object__'):
            return cls.__object__(arg, *args, **kwds)
        return arg

    def _is_hashable(cls, object):
        try:
            hash(object)
        except Exception:
            return False
        return True

    def _lookup_cached(cls, object):
        return cls.memory.get(object)

    def _register_instance(cls, object, self):
        if cls._is_hashable(object):
            cls.memory[object] = self

        if cls.primary:
            for base in cls.bases:
                if base is Object:
                    continue
                self.set(IsInstance[base], True)

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

    if False:
        def __dir__(cls):
            # for normal object, dir contains keys from:
            # * self.__dict__
            # * cls.__dict__ for cls in self.__class__.mro()
            # so if we want classes to behave like normal objects w.r.t. tab completion,
            # the __dir__ for a class produced by a metaclass should be:
            self_dir  = [object.__dir__(cls)]
            base_dirs = [object.__dir__(base) for base in cls.mro()]
            all_dirs  = self_dir + base_dirs
            dir = sum(all_dirs, [])
            dir = sorted(set(dir))
            return dir

    ######################################
    ### plumbing for getattr / setattr ###
    ######################################

    def instances(cls):
        instances = dict(cls.memory)
        for sub in sorted(cls.subs, key=lambda c: c.__qualname__):
            instances.update(sub.memory)
        return instances

    def similarities(cls, object):
        sims = cls.attention(object)
        keys = list(cls.memory.keys())
        return sort_dict(dict(zip(keys, sims)))

    def invert(cls, object):
        val = cls.project(object)
        for k, v in cls.memory.items():
            if v is val:
                return k
        raise KeyError("Projected value not found in memory")

    def project(cls, object):
        # HOLY FUCK THIS IS WAY BETTER FOR NON-ORTHOGONAL GENERALIZED BASES,
        # DON'T CHANGE IT, IT ABSOLUTELY CRUSHED THE ALTERNATIVES FOR THE
        # IMPORTANT (VERY-NON-ORTHOGONAL) CASE OF ORDINAL BASES.
        sims = cls.attention(object)
        idx  = int(jnp.argmax(sims))
        val  = list(cls.memory.values())[idx]
        return val

    def attention(cls, object):
        return slow.pre_attention_l1(cls, object)

    def __array__(cls):
        if len(cls.memory) == 0:
            raise Exception(f"Attribute {cls!r} doesn't yet have any instances.")
        vects = [o.__array__() for o in cls.memory.values()]
        return jnp.stack(vects, axis=0)

    ##########################
    ### self training code ###
    ##########################

    def perfect(cls):
        for name, self in cls.instances().items():
            if self.score() < 1.0:
                return False
        return True

    def learn(cls, **kwds):
        kwds.setdefault('threshold', 1.0)
        while not cls.perfect():
            for name, self in cls.instances().items():
                gradients.learn_until_score(self, **kwds)
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

    ###############################################
    ### context classes and __getitem__ support ###
    ###############################################

    primary = True


class Object(metaclass=Type):

    object = object

    def __new__(cls, *args, **kwds):

        if not args:
            object = None
        else:
            object = args[0]

        t = kwds.pop('t', None)

        if ALL_TYPES_NULLABLE:
            if not isinstance(object, cls.object) and object is not None:
                raise TypeError(f"object {object!r} is not of type {cls.object}")
        else:
            if not isinstance(object, cls.object):
                raise TypeError(f"object {object!r} is not of type {cls.object}")

        self = builtins.object.__new__(cls)
        self.object  = object
        self.attrs   = {}
        self.thought = Thought(t)
        self.type    = type(cls)
        return self

    def __init__(self, *args, **kwds):
        pass

    def __init_subclass__(cls, **kwds):
        super().__init_subclass__()
        for base in cls.bases:
            base.subs.append(cls)

    def think(self):
        return self.thought.think()

    def rethink(self, t):
        self.thought.rethink(t)
        return self

    def bits(self):
        bits = 0
        for attr in self.attrs:
            bits += jnp.log2(len(attr.memory))
        return bits.item()

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
        if self.attrs.get(attr) != value:
            self.setknow(attr, value)
            self.setfeel(attr, value)
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
        if isinstance(other, Object):
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
        if isinstance(value, cls):
            return value
        elif isinstance(value, Object):
            cls(value.unwrap())
            return value
        else:
            return cls(value)

    @classmethod
    def ensure_unthinkable(cls, object):
        if isinstance(object, Object):
            return object.unwrap()
        return object

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

    def __class_getitem__(cls, item):
        #cls = cls.contextfree()
        try: return cls.contexts[item]
        except KeyError: pass
        name = f"{cls.__name__}[{item}]"
        sub = Type(name, cls, primary=False, Item=item)
        cls.contexts[item] = sub
        return sub


def _forbid_pickling_of_dynamic_types():
    def _no_pickle(self):
        raise TypeError(
            "Dynamic think.Type objects must not be pickled directly"
        )
    Type.__reduce__ = _no_pickle
    Object.__reduce__ = _no_pickle

_forbid_pickling_of_dynamic_types()

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

class IsInstance(Object):
    object = bool
IsInstance(True)
IsInstance(False)


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

#################################
# System serialization support  #
#################################

from pathlib import Path
import pickle

def _all_types():
    """Return all known Types in the system."""
    seen = set()
    stack = [Object]
    out = []
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        stack.extend(t.subs)
    return out


def _serialize_system():
    """
    Produce a pure-Python snapshot of the entire system.
    """

    def assert_plain(obj):
        import pickle
        try:
            pickle.dumps(obj)
        except Exception as e:
            raise RuntimeError(
                f"Non-serializable object leaked into snapshot: {obj!r}"
            ) from e

    data = {
        "types": {},
        "objects": {},
        "contexts": {},
    }

    # Serialize types
    for T in _all_types():
        data["types"][T.name] = {
            "base": T.base.name if hasattr(T, "base") else None,
            "thought": jnp.array(T.think()),
        }

        if T.contexts:
            data["contexts"][T.name] = list(T.contexts.keys())

        # Serialize instances
        objs = {}
        for k, obj in T.memory.items():
            objs[repr(k)] = {
                "object": k,
                "thought": jnp.array(obj.think()),
                "attrs": {
                    attr.name: val.object
                    for attr, val in obj.attrs.items()
                },
            }
        data["objects"][T.name] = objs


    assert_plain(data)
    return data


def _deserialize_system(data):
    """
    Restore system state from snapshot.
    Assumes code defining Types already exists.
    """
    name_to_type = {T.name: T for T in _all_types()}

    # Phase 1: recreate contextual types
    for base_name, items in data.get("contexts", {}).items():
        base = name_to_type[base_name]
        for item in items:
            ctx = base[item]   # triggers __class_getitem__
            name_to_type[ctx.name] = ctx

    # Restore type thoughts
    for name, info in data["types"].items():
        T = name_to_type[name]
        T.rethink(info["thought"])

    # Restore objects
    for type_name, objs in data["objects"].items():
        T = name_to_type[type_name]
        for _, info in objs.items():
            obj = T(info["object"])
            obj.rethink(info["thought"])

            # Restore attributes
            for attr_name, raw_val in info["attrs"].items():
                attr = name_to_type[attr_name]
                obj.setknow(attr, attr(raw_val))
                obj.setfeel(attr, attr(raw_val))


def save_system(path):
    """
    Save the entire Think system to disk.
    """
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(_serialize_system(), f)


def load_system(path):
    """
    Load a previously saved Think system.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    _deserialize_system(data)

# thus ends the core
# all else is commentary

