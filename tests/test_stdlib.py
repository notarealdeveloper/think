#!/usr/bin/env python

import pytest

import think
from think import Letter, Word, Sentence
from think import Year, Month, Day, Date


@pytest.mark.train
def test_training_a_system_to_perfection():

    # Example of training a system until it can
    # perfectly encode all the knowledge it has

    think.thought_dim(1024)

    a = Word('supercalifragilisticexpialidocious')
    b = Word('antidisestablishementarianism')
    c = Word('cupcake')
    d = Word('hotdog')
    for w in ('hey', 'there', 'babycakes', 'how', 'are', 'you', 'doing'):
        Word(w)

    s = Sentence("This is a fucking brain")
    t = Sentence("How cool is this")
    u = Sentence("That bitch crazy")
    v = Sentence("Yo mama so fat when she skips a meal the market drops")

    d1 = Date('1987-01-14')
    d2 = Date('2021-07-25')
    d3 = Date('2022-05-06')

    D1 = Date("September 17th, 03")
    D2 = Date("April 11th")
    D3 = Date("February 22nd")

    think.learn()

    assert think.perfect()

    # what we get from a perfectly trained system

    for word in (a,b,c,d):
        for n, letter in enumerate(word.object):
            assert word.get(Letter[n]) == letter

    for sentence in (s,t,u,v):
        for n, word in enumerate(sentence.object):
            assert sentence.get(Word[n]) == word

    for sentence in (s,t,u,v):
        for n, word in enumerate(sentence.object):
            assert sentence.get(Word[n]) == word
            word = Word(word)
            for m, letter in enumerate(word.object):
                assert word.get(Letter[m]) == letter

    for date in (d1,d2,d3):
        year, month, day = date.object.split('-')
        assert date.get(Year) == year
        assert date.get(Month) == month
        assert date.get(Day) == day

    assert D1.get(Year)  == '2003'
    assert D1.get(Month) == '09'
    assert D1.get(Day)   == '17'

    assert D2.get(Month) == '04'
    assert D2.get(Day)   == '11'

    assert D3.get(Month) == '02'
    assert D3.get(Day)   == '22'

