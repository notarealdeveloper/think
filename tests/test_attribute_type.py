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
    assert type(Object(o)) is Object
    assert type(Bool(b))   is Bool
    assert type(Str(s))    is Str
    assert type(Int(i))    is Int
    assert type(Float(f))  is Float

def test_attribute_type_for_core_types():
    assert type(Object) is Type
    assert type(Bool)   is Type
    assert type(Str)    is Type
    assert type(Int)    is Type
    assert type(Float)  is Type

def test_attribute_type_for_core_meta():
    assert type(Type)       is type
    assert type(BoolType)   is type
    assert type(StrType)    is type
    assert type(IntType)    is type
    assert type(FloatType)  is type


def test_attribute_type_for_derived_objects_built_from_calling_core_meta():
    Alive   = BoolType('Alive')
    Name    = StrType('Name')
    Age     = IntType('Age')
    Size    = FloatType('Size')
    assert type(Alive(True))  is Alive
    assert type(Name('Dave')) is Name
    assert type(Age(42))      is Age
    assert type(Size(0.1))    is Size


def test_attribute_type_for_derived_objects_built_from_subclassing_core_types():
    class Alive(Bool):  pass
    class Name(Str):    pass
    class Age(Int):     pass
    class Size(Float):  pass
    assert type(Alive(True))  is Alive
    assert type(Name('Dave')) is Name
    assert type(Age(42))      is Age
    assert type(Size(0.1))    is Size

