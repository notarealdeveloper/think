#!/usr/bin/env python3

import pytest

from think import Bool, Int, Str, Float

def test_bool_idempotence():
    assert Bool(True) is Bool(True)

def test_str_idempotence():
    assert Str('cake') is Str('cake')

def test_int_idempotence():
    assert Int(3) is Int(3)

def test_float_idempotence():
    assert Float(42.69) is Float(42.69)

def test_bool_well_typed():
    with pytest.raises(TypeError):
        Bool(object())

def test_str_well_typed():
    with pytest.raises(TypeError):
        Str(object())

def test_int_well_typed():
    with pytest.raises(TypeError):
        Int(object())

def test_float_well_typed():
    with pytest.raises(TypeError):
        Float(object())

def test_distinct_values_of_the_same_type_are_distinct():
    assert Bool(True) != Bool(False)
    assert Str('cake') != Str('pie')
    assert Int(3) != Int(4)
    assert Float(42.69) != Float(69.42)

def test_that_the_type_of_instances_is_their_class():
    assert type(Bool(True)) is Bool
    assert type(Str('Dave')) is Str
    assert type(Int(42)) is Int
    assert type(Float(42.69)) is Float

