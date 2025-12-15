#!/usr/bin/env python

import pytest

import think
from think import Letter, Word, Sentence


@pytest.mark.train
def test_word():
    [Word(w) for w in ('Hello', 'nice', 'to', 'meet', 'you', 'goodbye')]
    w = Word("Hello")
    w.learn()
    assert w.get(Letter[0]) == 'H'
    assert w.get(Letter[1]) == 'e'
    assert w.get(Letter[2]) == 'l'
    assert w.get(Letter[3]) == 'l'
    assert w.get(Letter[4]) == 'o'

@pytest.mark.train
def test_sentence():
    s = Sentence("This is so cool")
    think.learn()
    assert Sentence.Item is Word
    assert Sentence.__object__('spam and eggs') == ('spam', 'and', 'eggs')
    assert s.get(Sentence.Item[0]) == 'This'
    assert s.get(Word[0]) == 'This'
    assert s.object == ('This', 'is', 'so', 'cool')
    assert s.__raw__ == 'This is so cool'

