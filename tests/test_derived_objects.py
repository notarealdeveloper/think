#!/usr/bin/env python3

import pytest

from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType

Alive   = BoolType('Alive')
Name    = StrType('Name')
Age     = IntType('Age')
Weight  = FloatType('Weight')

def test_metacall_derived_type_idempotence():
    assert Alive(True) is Alive(True)
    assert Name('Dave') is Name('Dave')
    assert Age(42) is Age(42)
    assert Weight(98.6) is Weight(98.6)

def test_metacall_derived_type_nontriviality():
    assert Alive(True) != Alive(False)
    assert Name('Dave') != Name('Bob')
    assert Age(42) != Age(43)
    assert Weight(42.69) != Weight(69.42)

def test_metacall_derived_objects_types():
    assert type(Alive(True))   is Alive
    assert type(Name('Dave'))  is Name
    assert type(Age(42))       is Age
    assert type(Weight(42.69)) is Weight


class Alive(Bool):  pass
class Name(Str):    pass
class Age(Int):     pass
class Size(Float):  pass

def test_class_derived_type_idempotence():
    assert Alive(True) is Alive(True)
    assert Name('Dave') is Name('Dave')
    assert Age(42) is Age(42)
    assert Weight(98.6) is Weight(98.6)

def test_class_derived_type_nontriviality():
    assert Alive(True) != Alive(False)
    assert Name('Dave') != Name('Bob')
    assert Age(42) != Age(43)
    assert Weight(42.69) != Weight(69.42)

def test_class_derived_objects_types():
    assert type(Alive(True))   is Alive
    assert type(Name('Dave'))  is Name
    assert type(Age(42))       is Age
    assert type(Weight(42.69)) is Weight
