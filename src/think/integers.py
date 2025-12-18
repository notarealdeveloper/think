#!/usr/bin/env python3


__all__ = [
    'Len',
    'Sign',
    'Digit',
    'Bit',
    'Decimal',
    'Binary',
]

import re
import jax.numpy as jnp

from think import Int

class Len(Int):
    pass

class Sign(Int):
    pass

class Numeral(Int):

    @classmethod
    def __object__(cls, m, *args, **kwds):
        return int(m)

    def __new__(cls, m, *args, **kwds):
        """ Ordinal basis implementation """
        n = getattr(cls, 'Item', 0)
        a = cls[n].think()
        b = cls[n+1].think()
        θ = cls.theta(m)
        t = a*jnp.cos(θ) + b*jnp.sin(θ)
        self = super().__new__(cls, m, *args, **kwds)
        return self

    @classmethod
    def theta(cls, m):
        M = len(cls.__instances__)
        return (m/M)*(jnp.pi/2)


class Digit(Numeral):
    __instances__ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class Bit(Numeral):
    __instances__ = [0, 1]


class Decimal(Int):
    def __init__(self, n):
        sign = +1 if n >= 0 else -1
        digits = str(abs(n))
        for slot, digit in enumerate(reversed(digits)):
            self.set(Digit[slot], int(digit))
        self.set(Len, len(digits))
        self.set(Sign, sign)


class Binary(Int):
    def __init__(self, n):
        digits = bin(n)[2:]
        for slot, bit in enumerate(reversed(digits)):
            self.set(Bit[slot], int(bit))
