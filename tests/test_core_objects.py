#!/usr/bin/env python3

import pytest

from think import Object, Bool, Int, Str, Float

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

def test_core_objects_type_is_core_type():
    assert type(Object(None))   is Object
    assert type(Bool(True))     is Bool
    assert type(Str('cake'))    is Str
    assert type(Int(42))        is Int
    assert type(Float(42.69))   is Float

def test_core_objects_with_distinct_python_objects_are_not_equal():
    assert Bool(True) != Bool(False)
    assert Str('cake') != Str('pie')
    assert Int(3) != Int(4)
    assert Float(42.69) != Float(69.42)

