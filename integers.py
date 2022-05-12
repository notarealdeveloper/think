#!/usr/bin/env python3


__all__ = [
    'Len',
    'Digit',
    'Bit',
    'Decimal',
    'Binary',
]

import jax.numpy as jnp

from think import Int

class Len(Int):
    pass

class Numeral(Int):

    @classmethod
    def __object__(cls, m):
        return int(m)

    def __new__(cls, m, t=None):
        """ Ordinal basis implementation """
        n = getattr(cls, 'Item', 0)
        a = cls[n].think()
        b = cls[n+1].think()
        θ = cls.theta(m)
        t = a*jnp.cos(θ) + b*jnp.sin(θ)
        self = super().__new__(cls, m, t=t)
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
        digits = str(n)
        for slot, digit in enumerate(reversed(digits)):
            digit = int(digit)
            print(f'self.set(Digit[{slot}], {digit})')
            self.set(Digit[slot], digit)


class Binary(Int):
    def __init__(self, n):
        digits = bin(n)[2:]
        for slot, bit in enumerate(reversed(digits)):
            self.set(Bit[slot], int(bit))


