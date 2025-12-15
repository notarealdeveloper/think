#!/usr/bin/env python3

__all__ = [
    'Add',
    'Sub',
    'Mul',
    'Div',
]

from think import slow
import think

class BinaryOperation:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __repr__(self):
        return f"{self.a.object} {self.op} {self.b.object}"

class Add(BinaryOperation):
    op = '+'
    def think(self):
        return slow.mix([self.a.think(), +self.b.think()])

class Sub(BinaryOperation):
    op = '-'
    def think(self):
        return slow.mix([self.a.think(), -self.b.think()])

class ScalarOperation:
    def __init__(self, a, b):
        if isinstance(a, think.Object):
            assert not isinstance(b, think.Object)
            object = a
            scalar = b
        elif isinstance(b, think.Object):
            assert not isinstance(a, think.Object)
            object = b
            scalar = a
        else:
            raise TypeError(a, b)
        self.o = object
        self.s = scalar

    def __repr__(self):
        return f"{self.s} {self.op} {self.o.object}"

class Mul(ScalarOperation):
    op = '*'
    def think(self):
        return self.o.think() * self.s

class Div(ScalarOperation):
    op = '/'
    def think(self):
        return self.o.think() / self.s

