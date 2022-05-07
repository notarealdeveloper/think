#!/usr/bin/env python3

from slow import cos
from think.core import Object, Str
from think import Char, Letter, Word, Sentence
from think import Digit, Digits
from think import Year, Month, Day
from think import Date
import think

think.thought_dim(1024)

# Words
a = Word('jababa')
b = Word('jabaga')
c = Word('banana')
d = Word('zyzygy')
assert cos(a, b) > cos(a, c) > cos(a, d)

w = Word('banana')
for n, letter in enumerate('banana'):
    assert letter == w.get(Letter[n])

# Digits
d = Digits('42069')
assert d.get(Digit[0]) == '9'
assert d.get(Digit[1]) == '6'
assert d.get(Digit[2]) == '0'
assert d.get(Digit[3]) == '2'
assert d.get(Digit[4]) == '4'

# Years
a = Year('98')
b = Year('1998')
c = Year('2098')
assert cos(a, b) > cos(a, c)

a = Year('08')
b = Year('1908')
c = Year('2008')
assert cos(a, b) < cos(a, c)

# Dates
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

# Sentences
s = Sentence('This is a brain')
assert s.get(Word[0]) == 'This'
assert s.get(Word[1]) == 'is'
assert s.get(Word[2]) == 'a'
assert s.get(Word[3]) == 'brain'

s = Sentence('How are you doing')
assert s.get(Word[0]) == 'How'
assert s.get(Word[1]) == 'are'
assert s.get(Word[2]) == 'you'
assert s.get(Word[3]) == 'doing'

s = Sentence('Whats going on')
assert s.get(Word[0]) == 'Whats'
assert s.get(Word[1]) == 'going'
assert s.get(Word[2]) == 'on'

s = 'This is a brain'
w = Sentence(s).get(Word[3])
assert w == 'brain'
assert Word(w).get(Letter[0]) == 'b'
assert Word(w).get(Letter[1]) == 'r'
assert Word(w).get(Letter[2]) == 'a'
assert Word(w).get(Letter[3]) == 'i'
assert Word(w).get(Letter[4]) == 'n'

a = Word(w).get(Letter[2])
i = Word(w).get(Letter[3])
a = Letter(a)
i = Letter(i)
#A = a.get(Letter.Uppercase)
#I = i.get(Letter.Uppercase)
#assert A == 'A'
#assert I == 'I'

