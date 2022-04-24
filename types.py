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

    def think(self):
        return self.thought.think()

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

    def __getitem__(self, key):
        return Getitem(self, key)

    def __setitem__(self, key, value):
        return Setitem(self, key, value)

    def get(self, attr):
        return Getattr(self, attr)

    def set(self, attr, value):
        return Setattr(self, attr, value)


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

    def __init__(self, name, base=object):
        self.name    = name
        self.base    = base
        self.object  = type(name, (self.base,), {})

    def __repr__(self):
        return f"{self.name}"


# ================================


class Integer(Type):
    def __init__(self, name, base=Int):
        if not issubclass(base, Number):
            raise TypeError(f"{name} has non-numeric base {base}")
        Type.__init__(self, name, base)
        self.origin = self.object.origin
        self.vector = self.object.vector

    def __call__(self, int):
        return self.object(int)

    def think(self, int=0):
        int = self._unwrap(int)
        s = self.object.sense(int)
        b = self.object.origin.think()
        w = self.object.vector.think()
        return b + s*w


class Floating(Type):
    def __init__(self, name, base=Float):
        if not issubclass(base, Number):
            raise TypeError(f"{name} has non-numeric base {base}")
        Type.__init__(self, name, base)
        self.origin = self.object.origin
        self.vector = self.object.vector

    def __call__(self, float):
        return self.object(float)

    def think(self, float=0.0):
        float = self._unwrap(float)
        s = self.object.sense(float)
        b = self.object.origin.think()
        w = self.object.vector.think()
        return b + s*w


class String(Type):
    def __init__(self, name, base=Str):
        Type.__init__(self, name, base)
        self.thought = self.object.thought
        self.memory = {}

    def __call__(self, str):
        return self.memory.get(str) or self.memory.setdefault(str, self.object(str))

    def think(self, str=None):
        str = self._unwrap(str)
        T = self.thought.think()
        if str is None:
            return T
        t = self.memory[str].think()
        return slow.mix([T, t])


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



