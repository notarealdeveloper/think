#!/usr/bin/env python3

# TODO:
# 1. define getattr and setattr
# 2. add tests showing a minimal working example


"""
    A machine learnable version of the python type system.
"""

import fast
import slow
import numbers
import jax.numpy as jnp
from think import Thought
from collections import defaultdict


class Object:

    """
        An .object you can .think() about.
    """

    type = object

    def __init__(self, object):
        self._check(object)
        self.object = object
        self.thought = Thought()
        self.attrs = {}

    def think(self):
        """
            This is not just self.thought.think().
            This is a computation derived from that thought plus our top down knowledge.
            That means it is the result of a computation that depends on learnables,
            and we can take a gradient of it with respect to them.
        """
        t = self.thought.think()
        t0 = t
        for attr, value in self.attrs.items():
            _pre_get_check_attr(attr)
            t = _hardset(t, attr, value)
        return t

    def rethink(self, t):
        return self.thought.rethink(t)

    def depends(self):
        return [self.thought]

    @classmethod
    def _check(cls, object):
        if not isinstance(object, cls.type):
            raise TypeError(f"not a {cls.type.__name__}: {object}")

    @classmethod
    def _unwrap(cls, object):
        while hasattr(object, 'object'):
            object = object.object
        return object

    @classmethod
    def least_base_type(cls, *objects):
        from functools import reduce
        from operator import and_
        from collections import Counter
        classes = [o.type for o in objects]
        return next(iter(reduce(and_, (Counter(cls.mro()) for cls in classes))))

    @classmethod
    def __init_subclass__(cls):
        cls.thought = Thought()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object!r})"

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Mul(self, other)

    def __rmul__(self, other):
        return Mul(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __eq__(self, other):
        if self is other:
            return True
        if (self.object != other.object):
            return False
        a = self.thought.think()
        b = other.thought.think()
        return jnp.allclose(a, b).item()

    def get(self, Attr):
        _pre_get_check_attr(Attr)
        if Attr not in self.attrs:
            feel = Attr.project(self)
            print(f"get: {self} asked for {Attr}, but no explicit knowledge exists. "
                  f"returning feeling: {feel}")
            return feel
        # here's where things get interesting
        value = self.attrs[Attr]

        feel  = Attr.project(self)
        know  = Attr.project(value)

        adeps = Attr.depends()
        odeps = self.depends()
        vdeps = value.depends()

        print(f"Unsupervised learning time bitches!")
        print(f"feel=know is the goal, and they're both computations with dependencies.")
        print(f"Go.")
        return {'feel':feel, 'know': know, 'odeps': odeps, 'vdeps': vdeps, 'adeps': adeps}

    def set(self, Attr, value):
        _pre_set_check_attr(Attr)
        value = _pre_set_promote_value(Attr, value)
        self._softset(Attr, value)
        # TODO: REMEMBER THIS:
        # If we do the hardset here (or if we ever do a hardset directly on a learnable),
        # we don't "absorb" the knowledge. (i.e., we don't *understand how to produce it*.)
        # instead, ONLY PERFORM HARDSETS ON THE OUTPUT OF COMPUTATIONS!
        # i.e., on things that are NOT directly "settable"
        # e.g., using the knowledge dictionary inside of get()
        # self._hardset(Attr, value)
        print(f"remember: set is being smart by not calling _hardset!")
        return self

    def softset(self, Attr, value):
        _pre_set_check_attr(Attr)
        value = _pre_set_promote_value(Attr, value)
        self._softset(Attr, value)
        return self

    def _softset(self, Attr, value):
        self.attrs[Attr] = value
        return self

    def __array__(self):
        return self.think()


### hard set implementation

def hardset(object, Attr, value):
    _pre_set_check_attr(Attr)
    value = _pre_set_promote_value(Attr, value)
    t_new = _hardset(object.think(), Attr, value)
    return t_new

def _hardset(t, Attr, value):
    # we need Attr to still be fancy here!
    t_in = Attr.project(t)
    v_in = Attr.project(value)
    t_out = t - t_in
    t_new = v_in + t_out
    return t_new

def _pre_get_check_attr(Attr):
    if not isinstance(Attr, Type):
        raise TypeError(f"get: Attr {Attr} is not an instance of Type")

def _pre_set_check_attr(Attr):
    if not isinstance(Attr, Type):
        raise TypeError(f"set: Attr {Attr} is not an instance of Type")

def _pre_set_promote_value(Attr, value):
    if not isinstance(value, Attr.object):
        value = Attr(value)
    return value

### end hard set implementation

class Str(Object):
    type = str


class List(Object):
    type = list


class Dict(Object):
    type = dict


class Tuple(Object):
    type = tuple


class Number(Object):

    type = numbers.Number

    def __init_subclass__(cls):
        cls.origin = Thought()
        cls.vector = Thought()

    def __init__(self, object):
        self._check(object)
        self.object = object
        # no per-instance thought

    def think(self):
        s = self.sense(self.object)
        b = self.origin.think()
        w = self.vector.think()
        return b + s*w

    @classmethod
    def sense(cls, object):
        return object


class Int(Number):
    type = int


class Float(Number):
    type = float


class Type(Object):
    type = type

    def __init__(cls, name, base=object):
        cls.name    = name
        cls.base    = base
        cls.object  = type(name, (cls.base,), {})
        cls.attrs   = {}

    def __call__(cls, object):
        self = cls.object(object)
        return self

    def __repr__(cls):
        return f"{cls.name}"

    def __hash__(self):
        return hash(self.object)

# ================================


class NumberType(Type):
    def __init__(self, name, base=Number):
        if not issubclass(base, Number):
            raise TypeError(f"{name} has non-numeric base {base}")
        Type.__init__(self, name, base)
        self.origin = self.object.origin
        self.vector = self.object.vector

    def think(self, number=0):
        number = self._unwrap(number)
        s = self.object.sense(number)
        b = self.object.origin.think()
        w = self.object.vector.think()
        return b + s*w

    def project(self, object):
        basis = [self.origin, self.vector]
        return slow.project(basis, object)

    def depends(self):
        return [self.origin, self.vector]


class Integer(NumberType):
    def __init__(self, name, base=Int):
        super().__init__(name, base)


class Floating(NumberType):
    def __init__(self, name, base=Float):
        super().__init__(name, base)


class EnumType(Type):
    def __init__(self, name, base):
        Type.__init__(self, name, base)
        self.thought = self.object.thought
        self.memory = {}

    def __call__(self, object):
        if object in self.memory:
            return self.memory[object]
        o = self.object(object)
        self.memory[object] = o
        return o

    def think(self, object=None):
        object = self._unwrap(object)
        T = self.thought.think()
        if object is None:
            return T
        t = self.memory[object].think()
        return slow.mix([T, t])

    def project(self, object):
        keys = list(self.memory.keys())
        vals = list(self.memory.values())
        return slow.project(vals, object)

    def depends(self):
        return [self.thought] + [o.thought for o in self.memory.values()]


class String(EnumType):
    def __init__(self, name, base=Str):
        super().__init__(name, base)


# ^ these guys are the ones who need invert methods!
# continue here tomorrow :)

# ================================


class Add(Object):

    def __init__(self, a, b):
        self.object = a.object + b.object
        self.type = Object.least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() + self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Sub(Object):

    def __init__(self, a, b):
        self.object = a.object - b.object
        self.type = Object.least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() - self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Mul(Object):

    def __init__(self, a, b):
        self.object = a.object * b.object
        self.type = Object.least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() * self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Div(Object):

    def __init__(self, a, b):
        self.object = a.object / b.object
        self.type = Object.least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() / self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")

# ================================

### TESTS!!!

a = Int(42)
b = Int(69)
c = a + b
assert a.type is int
assert b.type is int
assert jnp.allclose((a + b).think(), a.think() + b.think())
assert jnp.allclose((a - b).think(), a.think() - b.think())
assert type(c) is Add
assert c.type is int
assert c.a is a
assert c.b is b

Age = Integer('Age')
a = Age(42)
b = Age(69)
c = a + b
assert a.type is int
assert b.type is int
assert jnp.allclose((a + b).think(), a.think() + b.think())
assert jnp.allclose((a - b).think(), a.think() - b.think())
assert type(c) is Add
assert c.type is int
assert c.a is a
assert c.b is b

a = Str('Hello,')
b = Str(' world')
c = a + b
assert a.type is str
assert b.type is str
assert jnp.allclose((a + b).think(), a.think() + b.think())
assert type(c) is Add
assert c.type is str
assert c.a is a
assert c.b is b

a = Str('Hello,')
b = Str(' world')
c = a + b


#a = Int(42)
#b = Str('cake')
#c = SymbolicAdd(a, b)

Ticker = String('Ticker')
import pytest
with pytest.raises(TypeError):
    a = Ticker(42)

SPY = Ticker('SPY')
QQQ = Ticker('QQQ')
TLT = Ticker('TLT')
TLH = Ticker('TLH')
GLD = Ticker('GLD')
SLV = Ticker('SLV')


# test that we can compute the thoughts manually in the way we expect
Age = Integer('Age')
Age.think()
Age.think(42)
Age.think(Age(42))
a = (Age.origin.think() + 42*Age.vector.think())
b = Age.think(42)
assert jnp.allclose(a, b)


Ticker = String('Ticker')
Ticker('SPY')
Ticker.think()
Ticker.think('SPY')
Ticker.think(Ticker('SPY'))
a = (Ticker.thought.think() + Ticker('SPY').thought.think())/jnp.sqrt(2)
b = Ticker.think('SPY')
assert jnp.allclose(a, b)




# WHAT WE NEED TO STEAL FROM THE OLD BASIS TYPE:

if False:

    # Inversion for types that track their instances.
    def most_similar(Ty, Ob):
        t    = Ob.think()
        ts   = Type.stack()
        sims = fast.attention_l1(ts, t, norm=norm)
        idx  = int(jnp.argmax(sims))
        key  = self._keys[idx] # which "one" is most similar
        return key

    ###

    def coordinates(self, obj):
        t  = to_thought(obj)
        ts = self.stack()
        coords = jnp.squeeze(coordinates(ts, t), axis=-1)
        return Dict(zip(self.keys(), coords))

    def attention(self, obj, norm='l1'):
        t    = to_thought(obj)
        ts   = self.stack()
        sims = attention(ts, t, norm=norm)
        return Dict(zip(self.keys(), sims))

    def most_similar(self, obj, norm='l1'):
        t    = to_thought(obj)
        ts   = self.stack()
        sims = attention(ts, t, norm=norm)
        idx  = int(jnp.argmax(sims))
        key  = self._keys[idx]
        return key

    def solve(self, obj):
        t = to_thought(obj)
        return jnp.linalg.lstsq(self.stack().T, t)

    def hardset(self, obj, value):
        ts   = self.stack()
        t    = to_thought(obj)
        v    = to_thought(value)
        return hardset(ts, t, v)

    def softset(self, obj, value):
        ts   = self.stack()
        t    = to_thought(obj)
        v    = to_thought(value)
        return softset(ts, t, v)


Ticker  = String('Ticker')
SPY     = Ticker('SPY')
QQQ     = Ticker('QQQ')
TLT     = Ticker('TLT')

Age     = Integer('Age')
Sector  = String('Sector')
SEMI    = Sector('SEMI')
TECH    = Sector('TECH')
BANK    = Sector('BANK')

SPY.set(Age, 42)
assert SPY.attrs[Age] == Age(42)
SPY.set(Age, Age(69))
assert SPY.attrs[Age] == Age(69)

Ticker.get(Age)
# SPY.get(Age)

SPY.set(Sector, SEMI)
SPY.get(Sector)
