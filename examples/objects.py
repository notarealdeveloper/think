#!/usr/bin/env python3

from slow import cos
from think.core import Object, Str
from think import Char, Letter, Digit, Word
from think import Year, Month, Day, Date

# Word is a sequence of letters, until we learn more.
w1 = Word('jababa')
w2 = Word('jabaga')
cos(w1, w2)

a = Char('a')
b = Char('b')
cos(a, b)

a = Char('a')
b = Word('a')
cos(a, b)

a = Char('a')
b = Str('a')
cos(a, b)

a = Char('a')
b = Object('a')
cos(a, b)

a = Year('98')
b = Year('1998')
c = Year('2098')
assert cos(a, b) > cos(a, c)

a = Year('08')
b = Year('1908')
c = Year('2008')
assert cos(a, b) < cos(a, c)

a = Date('2020-01-14')
b = Date('January')
assert a.get(Year) == '2020'
assert a.get(Month) == '01'
assert a.get(Day) == '14'
assert b.get(Month) == 'January'

