#!/usr/bin/env python

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType

def test_attribute_object_for_core_objects():
    assert Object(None).object  is None
    assert Bool(True).object    is True
    assert Int(42).object       is 42
    assert Str('hi').object     == 'hi'
    assert Float(0.1).object    == 0.1

def test_attribute_object_for_core_types():
    assert Object.object is object
    assert Bool.object   is bool
    assert Str.object    is str
    assert Int.object    is int
    assert Float.object  is float

def test_attribute_object_for_core_meta():
    # Python has many built-in types, but only one built-in metaclass.
    # Since .object gives the corresponding *python* object of each
    # learnable object, the correct response to these questions is
    # the python builtin: 'type'
    assert Type.object      is type
    assert BoolType.object  is type
    assert StrType.object   is type
    assert IntType.object   is type
    assert FloatType.object is type

