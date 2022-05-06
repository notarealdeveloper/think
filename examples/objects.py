#!/usr/bin/env python3

from slow import cos
from think.core import Object, Str
from think import Char, Letter, Word, Sentence
from think import Digit, Digits
from think import Year, Month, Day
from think import Date
import think

# TODO: every Type needs to be a memory type, so it can be used as an attribute.
# Maybe don't enforce this in the core, but it should be the default everywhere
# else. Also, attributes can try to compress themselves by removing unnecessary
# basis vectors if they can express some thoughts in terms of others.

think.thought_dim(1024)

# Word is a sequence of letters, until we learn more about it.
a = Word('jababa')
b = Word('jabaga')
c = Word('banana')
d = Word('zyzygy')
assert cos(a, b) > cos(a, c) > cos(a, d)

w = Word('banana')
for n, letter in enumerate(reversed('banana')):
    assert w.get(Letter[n]) == letter

d = Digits('42069')
assert d.get(Digit[0]) == '9'
assert d.get(Digit[1]) == '6'
assert d.get(Digit[2]) == '0'
assert d.get(Digit[3]) == '2'
assert d.get(Digit[4]) == '4'

a = Year('98')
b = Year('1998')
c = Year('2098')
assert cos(a, b) > cos(a, c)

a = Year('08')
b = Year('1908')
c = Year('2008')
assert cos(a, b) < cos(a, c)

a = Date('January')
assert a.get(Month) == 'January'

a = Date('2021-07-25')
assert a.get(Day) == '25'
assert a.get(Month) == '07'
assert a.get(Year) == '2021'
assert a.get(Year.Millenium) == '2'
assert a.get(Year.Century) == '0'
assert a.get(Year.Decade) == '2'
assert a.get(Year.Year) == '1'

for n, digit in enumerate(reversed('2021')):
    assert a.get(Digit[n]) == digit



