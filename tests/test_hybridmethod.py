#!/usr/bin/env python3

import pytest

from think import Thought, new_thought
from think.internals import hybridmethod

def run_think_methods(a, A, descriptor):

    B = None
    b = None

    class O:
        t = Thought(A)
        def __init__(self):
            self.t = Thought(a)

        @descriptor
        def think(self):
            return self.t.think()

    o = O()

    try:    b = o.think()
    except: pass

    try:    B = O.think()
    except: pass

    return b, B

def test_classmethod_doesnt_work():
    a, b = new_thought(), new_thought()
    A, B = run_think_methods(a, b, classmethod)
    assert (a is not A) or (b is not B)

def test_staticmethod_doesnt_work():
    a, b = new_thought(), new_thought()
    A, B = run_think_methods(a, b, classmethod)
    assert (a is not A) or (b is not B)

def test_instancemethod_doesnt_work():
    a, b = new_thought(), new_thought()
    A, B = run_think_methods(a, b, lambda x: x)
    assert (a is not A) or (b is not B)

def test_hybridmethod_works():
    a, b = new_thought(), new_thought()
    A, B = run_think_methods(a, b, hybridmethod)
    assert (a is A) and (b is B)

