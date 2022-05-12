#!/usr/bin/env python3

__all__ = [
    # bits
    'Len',
    'Bit',
    'Binary',
    # strings
    'Char',
    'Letter',
    'Word',
    # numerals
    'Digit',
    'Digits',
    # time
    'Year',
    'Month',
    'Day',
    'Weekday',
    'Date',
    # collections
    'Item',
    'Key',
    'Value',
    'List',
    'Tuple',
    'Set',
    'Dict',
    # data structures
    'After',
    'Before',
    'LinkedList',
    'DoublyLinkedList',
    'Sentence',
]

import re
import string
import datetime

import slow
from think import Bool, Str, Int
from think import Object, Type


# Bits
class Len(Int):
    pass

class Bit(Str):
    __create__ = ['0', '1']

class Binary(Int):
    def __init__(self, n):
        encoding = bin(n)[2:]
        for n, bit in enumerate(reversed(encoding)):
            self.set(Bit[n], bit)
        self.set(Len, len(encoding))


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

    @classmethod
    def __object__(cls, arg):
        if isinstance(arg, int) and 1 <= arg <= 7:
            object = cls.INT_TO_NAME[arg]
        elif isinstance(arg, str) and arg in '1234567':
            object = cls.INT_TO_NAME[int(arg)]
        elif isinstance(arg, str) and arg in cls.NAME_TO_INT:
            object = arg
        else:
            raise ValueError(arg)
        return object


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

class Item(Object):
    """ For objects contained within other objects,
        like the elements of a list, tuple, or set.
    """
    pass

class Key(Object):
    pass

class Value(Object):
    pass


class Sequence(Object):
    Item = Item

    def __init__(self, seq):
        for n in range(len(seq)):
            self.set(self.Item[n], seq[n])
        self.set(Len, len(seq))

    def __getitem__(self, n):
        return self.get(self.Item[n])

    def __len__(self):
        return self.get(Len)


class Tuple(Sequence):
    object = tuple

class List(Sequence):
    object = list

class Set(Sequence):
    object = set


class Dict(Sequence):
    object = dict
    Key   = Key
    Value = Value

    def __init__(self, dict):
        for n, (k,v) in enumerate(dict.items()):
            self.set(self.Item[k], v)
            self.set(self.Key[n], k)
            self.set(self.Value[n], v)
        self.set(Len, len(dict))


# Linked Lists

class After(Object):
    pass

class Before(Object):
    pass

class Head(Object):
    pass

class LinkedList(Object):
    object = list
    Head = Head
    After = After
    Before = Before

    def __init__(self, list):
        items = [*list, None]
        self.set(self.Head, items[0])
        for this, next in zip(items[:-1], items[+1:]):
            self.set(self.After[this], next)


class DoublyLinkedList(LinkedList):

    def __init__(self, list):
        self.set(self.Head, None)
        items = [None, *list, None]
        for this, next in zip(items[:-1], items[+1:]):
            self.set(self.After[this], next)
            self.set(self.Before[next], this)

# Sentences

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

