#!/usr/bin/env python3

__all__ = [
    'Digits',
    'Bits',
    'Char',
    'Letter',
    'Word',
    'Sentence',
]

import string

from think import Bool, Str
from think import Digit, Bit
from think import List


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

class Char(Str):
    def __init__(self, chr):
        if len(chr) > 1:
            raise TypeError(f"Not a char: {chr!r}")


class Letter(Char):

    LETTERS   = set(string.ascii_letters)
    LOWERCASE = set(string.ascii_lowercase)
    UPPERCASE = set(string.ascii_uppercase)

    LOWER_TO_UPPER = dict(zip(string.ascii_lowercase, string.ascii_uppercase))
    UPPER_TO_LOWER = dict(zip(string.ascii_uppercase, string.ascii_lowercase))

    def __init__(self, letter):
        if letter not in self.LETTERS:
            raise TypeError(f"Not a letter: {letter!r}")

        return
        if letter in self.UPPERCASE:
            self.set(self.IsLowercase, False)
            self.set(self.IsUppercase, True)
            self.set(self.Lowercase, self.UPPER_TO_LOWER[letter])

        if letter in self.LOWERCASE:
            self.set(self.IsUppercase, False)
            self.set(self.IsLowercase, True)
            self.set(self.Uppercase, self.LOWER_TO_UPPER[letter])

    class IsUppercase(Bool): pass
    class IsLowercase(Bool): pass

    class Uppercase(Char): pass
    class Lowercase(Char): pass


class Word(Str):
    def __init__(self, word):
        for n, letter in enumerate(word):
            self.set(Letter[n], letter)


class Sentence(List[Word]):

    @classmethod
    def __object__(cls, str):
        try:
            import nltk
            words = super().__object__(nltk.tokenize.word_tokenize(str))
        except:
            words = str.split(' ') # poor man's tokenizer, if we don't have nltk
        return words

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__raw__})"

