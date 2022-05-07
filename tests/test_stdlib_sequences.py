#!/usr/bin/env python3

import pytest

import think
from think import List, Tuple, Set, Dict
from think import LinkedList, DoublyLinkedList, After, Before
from think import Word, Sentence


def test_sequences():

    t = Tuple(('a', 42, 3+2j))
    l = List(['a', 42, 3+2j])
    d = Dict({'name': 'dave', 'age': 42})

    think.learn()

    assert l.get(List.Item[0]) == 'a'
    assert l.get(List.Item[1]) == 42
    assert l.get(List.Item[2]) == 3+2j

    assert t.get(Tuple.Item[0]) == 'a'
    assert t.get(Tuple.Item[1]) == 42
    assert t.get(Tuple.Item[2]) == 3+2j

    assert d.get(Dict.Key[0]) == 'name'
    assert d.get(Dict.Value[0]) == 'dave'
    assert d.get(Dict.Item['name']) == 'dave'
    assert d.get(Dict.Key[1]) == 'age'
    assert d.get(Dict.Value[1]) == 42
    assert d.get(Dict.Item['age']) == 42


def test_sentences():

    s = Sentence("This is so cool")

    think.learn()

    assert Sentence.Item is Word
    assert Sentence.__object__('spam and eggs') == ['spam', 'and', 'eggs']
    assert s.get(Sentence.Item[0]) == 'This'
    assert s.get(Word[0]) == 'This'
    assert s.object == ['This', 'is', 'so', 'cool']
    assert s.__raw__ == 'This is so cool'


def test_linked_list():

    list = ['This', 'is', 'a', 'linked', 'list']
    ll = LinkedList(list)

    think.learn()

    this = ll.get(ll.Head)
    walk = []
    while this is not None:
        walk.append(this)
        this = ll.get(ll.After[this])
    assert walk == list


def test_doubly_linked_list():

    list = ['This', 'is', 'a', 'doubly', 'linked', 'list']
    ld = DoublyLinkedList(list)

    ld.learn() # different

    forward = []
    head = ld.get(ld.Head)
    this = ld.get(ld.After[head])
    while this is not None:
        forward.append(this)
        this = ld.get(ld.After[this])

    backward = []
    head = ld.get(ld.Head)
    this = ld.get(ld.Before[head])
    while this is not None:
        backward.append(this)
        this = ld.get(ld.Before[this])

    assert forward  == list
    assert backward == list[::-1]

