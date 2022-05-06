#!/usr/bin/env python3

__all__ = [
    'Char',
    'Letter',
    'Digit',
    'Digits',
    'Word',
    'Year',
    'Month',
    'Day',
    'Date',
    'Sentence',
]

import slow
import string
from think import Bool, Str, Type, Thought


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


# Words

class Word(Str):
    def __init__(self, letters):
        for n, letter in enumerate(letters):
            self.set(Letter[n], letter)


# Digits

class Digit(Char):
    DIGITS = set(string.digits)
    def __init__(self, digit):
        if digit not in self.DIGITS:
            raise TypeError(f"Not a digit: {digit!r}")


class Digits(Str):

    def __init__(self, digits):
        if not digits.isnumeric():
            raise TypeError(f"Not a numeric string: {digits!r}")
        self.digits = list(digits)
        for n, digit in enumerate(reversed(digits)):
            self.set(Digit[n], digit)


# Dates and Times

class Year(Str):

    Millenium = Digit[3]
    Century   = Digit[2]
    Decade    = Digit[1]
    Year      = Digit[0]

    def __init__(self, year):
        if not year.isnumeric():
            raise TypeError(f"Not a year: {year!r}")
        if len(year) == 4:
            pass
        elif len(year) == 2:
            year = f"20{year}" if year[0] in '012' else f"19{year}"
        else:
            raise TypeError(year)
        digits = list(year)
        for n, digit in enumerate(reversed(digits)):
            self.set(Digit[n], digit)


class Month(Str):
    names = {
        'january', 'february', 'march',
        'april', 'may', 'june', 'july',
        'august', 'september', 'october',
        'november', 'december',
    }
    @classmethod
    def is_month_name(cls, str):
        return str.lower() in cls.names


class Day(Str):
    pass


class Date(Str):
    def __init__(self, date):
        self.date = date
        if len(date) == 4 and date.isnumeric():
            attrs = {Year: date}
        elif Month.is_month_name(date):
            attrs = {Month: date}
        elif '-' in date:
            year, month, day = date.split('-')
            attrs = {Year: year, Month: month, Day: day}
        else:
            raise ValueError(date)
        self.attrs = attrs
        for attr, value in self.attrs.items():
            self.setfeel(attr, value)


# Sentences

class Sentence(Str):
    def __init__(self, sentence):
        self.words = sentence.split(' ')
        for n, word in enumerate(self.words):
            self.set(Word[n], word)

