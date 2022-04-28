#!/usr/bin/env python3

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType

o = object()
b = True
i = 42
s = 'hi'
f = 0.05

def test_attribute_type_for_core_objects():
    assert Object(o).type is Object
    assert Bool(b).type   is Bool
    assert Str(s).type    is Str
    assert Int(i).type    is Int
    assert Float(f).type  is Float

def test_attribute_type_for_core_types():
    assert Object.type is Type
    assert Bool.type   is Type
    assert Str.type    is Type
    assert Int.type    is Type
    assert Float.type  is Type

def test_attribute_type_for_core_meta():
    assert Type.type       is Type
    assert BoolType.type   is Type
    assert StrType.type    is Type
    assert IntType.type    is Type
    assert FloatType.type  is Type


def test_attribute_type_for_derived_objects_of_():
    Alive   = BoolType('Alive')
    Name    = StrType('Name')
    Age     = IntType('Age')
    Size    = FloatType('Size')
    assert Alive(True).type  is Alive
    assert Name('Dave').type is Name
    assert Age(42).type      is Age
    assert Size(0.1).type    is Size


def test_attribute_type_for_derived_objects_built_from_subclassing_core_types():
    class Alive(Bool):  pass
    class Name(Str):    pass
    class Age(Int):     pass
    class Size(Float):  pass
    assert Alive(True).type  is Alive
    assert Name('Dave').type is Name
    assert Age(42).type      is Age
    assert Size(0.1).type    is Size

