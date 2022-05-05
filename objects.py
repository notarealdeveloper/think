#!/usr/bin/env python3

__all__ = [
    'Char',
    'Letter',
    'Digit',
    'Word',
    'Year',
    'Month',
    'Day',
    'Date',
]

import slow
import string
from think import Str, EnumType
from think import Thought


class Char(Str, metaclass=EnumType):
    def __init__(self, chr):
        if len(chr) > 1:
            raise ValueError(f"Not a char: {chr!r}")

class Letter(Char):
    def __init__(self, letter):
        if letter not in string.ascii_letters:
            raise ValueError(f"Not a letter: {letter!r}")

class Digit(Char):
    def __init__(self, digit):
        if digit not in string.digits:
            raise ValueError(f"Not a digit: {digit!r}")

class Word(Str, metaclass=EnumType):
    def __init__(self, word):
        super().__init__(word)
        letters = [Letter(c).think() for c in word]
        t = slow.mix(letters)
        self.rethink(t)


class Year(Str, metaclass=EnumType):
    class M(Digit): pass
    class C(Digit): pass
    class D(Digit): pass
    class Y(Digit): pass
    def __init__(self, year):
        if len(year) == 4:
            m, c, d, y = year
        elif len(year) == 2:
            d, y = year
            if d in ['0', '1', '2']:
                m, c = '20'
            else:
                m, c = '19'
        else:
            raise ValueError(year)
        attrs = {
            self.M: m,
            self.C: c,
            self.D: d,
            self.Y: y,
        }
        #thoughts = [attr(value).thought for attr, value in attrs.items()]
        #self.thought = Thought(slow.mix(thoughts))
        self.attrs = attrs
        for attr, value in self.attrs.items():
            self.setfeel(attr, value)


class Month(Str, metaclass=EnumType):
    names = {
        'january', 'february', 'march',
        'april', 'may', 'june', 'july',
        'august', 'september', 'october',
        'november', 'december',
    }
    @classmethod
    def is_month_name(cls, str):
        return str.lower() in cls.names


class Day(Str, metaclass=EnumType):
    pass


class Date(Str):
    # not an enumtype.
    # the set of dates is too big to remember them all.
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
        #thoughts = [attr(value).thought for attr, value in attrs.items()]
        #self.thought = Thought(slow.mix(thoughts))
        for attr, value in self.attrs.items():
            self.setfeel(attr, value)

