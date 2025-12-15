#!/usr/bin/env python3

__all__ = [
    'Item',
    'Key',
    'Value',
    'List',
    'Tuple',
    'Set',
    'Dict',
    'After',
    'Before',
    'LinkedList',
    'DoublyLinkedList',
]

from think import Object
from think import Len

# List, Tuple, Set

class Item(Object):
    """ For objects contained within other objects,
        like the elements of a list, tuple, or set. """
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

    @classmethod
    def invert(cls, t):
        items = [cls.Item[n].invert(t) for n in reversed(range(Len.invert(t)))]
        return items


class List(Sequence):
    object = list

class Tuple(Sequence):
    object = tuple

class Set(Sequence):
    object = set


# Dict

class Key(Object):
    pass


class Value(Object):
    pass


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


