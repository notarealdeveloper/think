#!/usr/bin/env python3

__all__ = [
    'Char',
    'Letter',
    'Digit',
    'Digits',
    'Word',
    'WordAfter',
    'WordBefore',
    'Year',
    'Month',
    'Day',
    'Weekday',
    'Date',
    'Sentence',
]

import re
import string
import datetime

import slow
from think import Bool, Str
from think import Object, Type

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

# Words

class Word(Str):
    def __init__(self, word):
        for n, letter in enumerate(word):
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
    pass

class Day(Str):
    pass

class Weekday(Str):
    INT_TO_NAME =  {1:'Monday', 2:'Tuesday', 3:'Wednesday',
                    4:'Thursday', 5:'Friday', 6:'Saturday',
                    7:'Sunday'}
    NAME_TO_INT = {v:k for k,v in INT_TO_NAME.items()}
    def __new__(cls, arg):
        if isinstance(arg, int) and 1 <= arg <= 7:
            object = cls.INT_TO_NAME[arg]
        elif isinstance(arg, str) and arg in '1234567':
            object = cls.INT_TO_NAME[int(arg)]
        elif isinstance(arg, str) and arg in cls.NAME_TO_INT:
            object = arg
        else:
            raise ValueError(arg)
        self = super().__new__(cls, object)
        return self


class Date(Str):

    def __init__(self, date):
        data = self.parse(date)
        if year := data.get('year'):
            self.set(Year, year)
        if month := data.get('month'):
            self.set(Month, month)
        if day := data.get('day'):
            self.set(Day, day)
        if weekday := data.get('weekday'):
            self.set(Weekday, weekday)

    @classmethod
    def date_object_to_dict(cls, date):
        d = {k: getattr(date, k) for k in ('year', 'month', 'day', 'weekday')}
        if callable(d['weekday']): # for 2/3 methods, it's a function
            d['weekday'] = d['weekday']()
        d = {k:str(v) for k,v in d.items()}
        length = {'year': 4, 'month': 2, 'day': 2, 'weekday': 1}
        for k in d:
            d[k] = d[k].zfill(length[k])
        return d

    @classmethod
    def suspicious_year(cls, input, output):
        this_year = datetime.date.today().year
        return  output['year'] == this_year \
        and     this_year not in re.findall(r'\d{4}', input)

    @classmethod
    def parse_with_timestring(cls, input):
        import timestring
        date = timestring.Date(input)
        output = cls.date_object_to_dict(date)
        if cls.suspicious_year(input, output):
            # timestring gives the present year when no year is found
            output['year'] = None
            output['weekday'] = None
        return output

    @classmethod
    def parse_with_dateparser(cls, input):
        import dateparser
        date = dateparser.parse(input)
        output = cls.date_object_to_dict(date)
        if cls.suspicious_year(input, output):
            # dateparser gives the present year when no year is found
            output['year'] = None
            output['weekday'] = None
        return output

    @classmethod
    def parse_with_datetime_module(cls, input):
        date = datetime.date.fromisoformat(input)
        output = cls.date_object_to_dict(date)
        return output

    def parse(cls, input):
        try:
            return cls.parse_with_timestring(input)
        except:
            pass
        try:
            return cls.parse_with_dateparser(input)
        except:
            pass
        try:
            return cls.parse_with_datetime_module(input)
        except:
            pass
        raise ValueError(date)


# Sequences

class After(Object):
    pass

class Before(Object):
    pass

class WordAfter(Word, After):
    pass

class WordBefore(Word, Before):
    pass

# Sentences

class Sentence(Object): # this should be a List[Str]
    def __init__(self, sentence):
        self.object = tuple(sentence.split(' '))
        self.sentence = sentence
        for n, word in enumerate(self.object):
            self.set(Word[n], word)
        for prev, word in zip(self.object[:-1], self.object[+1:]):
            self.set(WordAfter[prev], word)
            self.set(WordBefore[word], prev)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.sentence})"

