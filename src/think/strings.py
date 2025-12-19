#!/usr/bin/env python3

__all__ = [
    'Digits',
    'Bits',
    'Letter',
    'Word',
    'Sentence',
]

import re
import string

from think import Bool, Str
from think import Digit, Bit
from think import Tuple


# Numbers

class Digits(Str):
    """
        Digits
        ======
        A base10 number implementation using Str.
        This type works the same as Decimal, but it
        takes a string rather than an int.
        In data type like Year, we often want the
        leading zeros in instances like 0042 to have
        a specific meaning.
    """
    def __init__(self, digits):
        for slot, digit in enumerate(reversed(digits)):
            self.set(Digit[slot], int(digit))


class Bits(Str):
    def __init__(self, bits):
        if 'b' in bits:
            assert bits[:2] == '0b'
            bits = bits[2:]
        for slot, bit in enumerate(reversed(bits)):
            self.set(Bit[slot], int(bit))


# Letters

#class Char(Str):
#    def __init__(self, chr):
#        if len(chr) > 1:
#            raise TypeError(f"Not a char: {chr!r}")
#

#class Letter(Char):
#
#    LETTERS = set(string.ascii_letters)
#
#    def __init__(self, letter):
#        if letter not in self.LETTERS:
#            raise TypeError(f"Not a letter: {letter!r}")


#class Letters(Str):
#    def __init__(self, word):
#        for n, letter in enumerate(word):
#            self.set(Letter[n], letter)

class Letter(Str):
    def __init__(self, chr):
        if len(chr) > 1:
            raise TypeError(f"Not a letter: {chr!r}")

class Word(Str):
    def __init__(self, word):
        for n, letter in enumerate(word):
            self.set(Letter[n], letter)


class Sentence(Tuple[Word]):

    @classmethod
    def __object__(cls, str):
        words = re.findall(r'"|\'|\w+(?:\'\w+)?|\S', str)
        return tuple(words)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__raw__})"

