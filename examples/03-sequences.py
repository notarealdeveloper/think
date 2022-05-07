#!/usr/bin/env python3

import pytest

from think import List, Tuple, Set, Dict
from think import LinkedList, DoublyLinkedList, After, Before
from think import Word, Sentence

l = List(['a', 42, 3+2j])
assert l.get(List.Item[0]) == 'a'
assert l.get(List.Item[1]) == 42
assert l.get(List.Item[2]) == 3+2j

t = Tuple(('a', 42, 3+2j))
assert t.get(Tuple.Item[0]) == 'a'
assert t.get(Tuple.Item[1]) == 42
assert t.get(Tuple.Item[2]) == 3+2j

d = Dict({'name': 'dave', 'age': 42})
assert d.get(Dict.Key[0]) == 'name'
assert d.get(Dict.Value[0]) == 'dave'
assert d.get(Dict.Item['name']) == 'dave'
assert d.get(Dict.Key[1]) == 'age'
assert d.get(Dict.Value[1]) == 42
assert d.get(Dict.Item['age']) == 42

assert Sentence.Item is Word
assert Sentence.__object__('spam and eggs') == ['spam', 'and', 'eggs']

s = Sentence("This is so cool")
assert s.get(Sentence.Item[0]) == 'This'
assert s.get(Word[0]) == 'This'
assert s.object == ['This', 'is', 'so', 'cool']
assert s.__raw__ == 'This is so cool'


list = ['This', 'is', 'a', 'linked', 'list']
l = LinkedList(list)
this = l.get(l.Head)
walk = []
while this is not None:
    walk.append(this)
    this = l.get(l.After[this])
assert walk == list

list = ['This', 'is', 'a', 'linked', 'list']
l = DoublyLinkedList(list)
forward = []
head = l.get(l.Head)
this = l.get(l.After[head])
while this is not None:
    forward.append(this)
    this = l.get(l.After[this])

backward = []
head = l.get(l.Head)
this = l.get(l.Before[head])
while this is not None:
    backward.append(this)
    this = l.get(l.Before[this])

assert forward  == list
assert backward == list[::-1]

