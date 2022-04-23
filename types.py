#!/usr/bin/env python3

"""
    A machine learnable version of the python type system.
"""

import numbers
from . import Thought


LIFTS = {}

class Object:

    type = object

    def __init__(self, object):
        if not isinstance(object, self.type):
            raise TypeError(f"not a {self.type.__name__}: {object}")
        self.object = object
        self.thought = Thought()

    def think(self):
        return self.thought.think()

    def rethink(self, t):
        return self.thought.rethink(t)

    def depends(self):
        return [self.thought]

    @staticmethod
    def least_base_type(*objects):
        from functools import reduce
        from operator import and_
        from collections import Counter
        classes = [o.type for o in objects]
        return next(iter(reduce(and_, (Counter(cls.mro()) for cls in classes))))

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object})"

    def __init_subclass__(cls):
        if cls.type not in LIFTS:
            LIFTS[cls.type] = cls

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
        cls.w = Thought()
        super().__init_subclass__()

    def __init__(self, object):
        if not isinstance(object, self.type):
            raise TypeError(f"not a {self.type.__name__}: {object}")
        self.object = object

    def think(self):
        s = self.sense(self.object)
        w = self.w.think()
        return s*w

    def depends(self):
        return [self.__class__.w]

    def sense(self, number):
        return number


class Int(Number):
    type = int


class Float(Number):
    type = float


class Type(Object):
    type = type

    def __init__(self, name, base=object):
        dict = {}
        self.name    = name
        self.base    = base
        self.Base    = LIFT_TYPE[base]
        self.object  = type(name, (self.base,), dict)
        self.Object  = type(name, (self.Base,), dict)
        self.thought = Thought()

    def __call__(self, o, **kwds):
        if not isinstance(o, self.base):
            raise TypeError(f"not an instance of {self.base.__name__}: {o!r}")
        obj = self.object(o, **kwds)
        Obj = self.Object(obj)
        return Obj

    def __repr__(self):
        return f"{self.name}"


class Add(Object):

    def __init__(self, a, b):
        self.object = a.object + b.object
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() + self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Sub(Object):

    def __init__(self, a, b):
        self.object = a.object - b.object
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() - self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Mul(Object):

    def __init__(self, a, b):
        self.object = a.object * b.object
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() * self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Div(Object):

    def __init__(self, a, b):
        self.object = a.object / b.object
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() / self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class SymbolicAdd(Object):

    def __init__(self, a, b):
        self.object = ('+', a, b)
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() + self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")

    def __repr__(self):
        op, a, b = self.object
        return f"{self.__class__.__name__}({a.object} {op} {b.object})"


class SymbolicSub(Object):

    def __init__(self, a, b):
        self.object = ('-', a, b)
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() - self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")

    def __repr__(self):
        op, a, b = self.object
        return f"{self.__class__.__name__}({a.object} {op} {b.object})"


class SymbolicMul(Object):

    def __init__(self, a, b):
        self.object = ('*', a, b)
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() * self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")

    def __repr__(self):
        op, a, b = self.object
        return f"{self.__class__.__name__}({a.object} {op} {b.object})"


class SymbolicDiv(Object):

    def __init__(self, a, b):
        self.object = ('/', a, b)
        self.type = least_base_type(a, b)
        self.a = a
        self.b = b

    def think(self):
        return self.a.think() / self.b.think()

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")

    def __repr__(self):
        op, a, b = self.object
        return f"{self.__class__.__name__}({a.object} {op} {b.object})"


class Getattr(Object):

    def __init__(self, a, b):
        self.object = getattr(a.object, b.object)
        self.a = a
        self.b = b

    def think(self):
        raise NotImplementedError(f"Use getattr in thought space to implement this.")

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


class Setattr(Object):

    def __init__(self, a, b, c):
        setattr(a.object, b.object, c.object)
        self.object = a
        self.a = a
        self.b = b
        self.c = c

    def think(self):
        raise NotImplementedError(f"Use setattr in thought space to implement this.")

    def rethink(self):
        raise NotImplementedError(f"Can't rethink about {self.__class__.__name__} object.")


