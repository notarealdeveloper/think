#!/usr/bin/env python3

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType

class Boolean(BoolType):
    pass

class String(StrType):
    pass

class Integer(IntType):
    pass

class Floating(FloatType):
    pass

def test_idemopotence_for_derived_types():
    assert Boolean('Alive') == Boolean('Alive')
    assert String('Name')   == String('Name')
    assert Integer('Age')   == Integer('Age')
    assert Floating('Size') == Floating('Size')

def test_type_attribute_for_derived_types():
    assert Boolean.object  is type
    assert String.object   is type
    assert Integer.object  is type
    assert Floating.object is type

def test_base_attribute_for_derived_types():
    assert Boolean.base  is Bool
    assert String.base   is Str
    assert Integer.base  is Int
    assert Floating.base is Float

