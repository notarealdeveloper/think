#!/usr/bin/env python3

import pytest

from think.types import meta
from think.types import BoolType, StrType, IntType, FloatType

Alive   = BoolType('Alive')
Name    = StrType('Name')
Age     = IntType('Age')
Weight  = FloatType('Weight')


def test_derived_bool_idempotence():
    assert Alive(True) is Alive(True)

def test_derived_str_idempotence():
    assert Name('Dave') is Name('Dave')

def test_derived_int_idempotence():
    assert Age(42) is Age(42)

def test_derived_float_idempotence():
    assert Weight(98.6) is Weight(98.6)

def test_derived_bool_is_well_typed():
    with pytest.raises(TypeError):
        Alive(object())

def test_derived_str_is_well_typed():
    with pytest.raises(TypeError):
        Name(object())

def test_derived_int_is_well_typed():
    with pytest.raises(TypeError):
        Age(object())

def test_derived_float_is_well_typed():
    with pytest.raises(TypeError):
        Weight(object())

def test_distinct_values_of_the_same_derived_type_are_distinct():
    assert Alive(True) != Alive(False)
    assert Name('Dave') != Name('Bob')
    assert Age(42) != Age(43)
    assert Weight(42.69) != Weight(69.42)

def test_that_the_type_of_derived_instances_is_their_class():
    assert meta(Alive(True)) is Alive
    assert meta(Name('Dave')) is Name
    assert meta(Age(42)) is Age
    assert meta(Weight(42.69)) is Weight


