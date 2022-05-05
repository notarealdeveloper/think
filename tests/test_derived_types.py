#!/usr/bin/env python3

import pytest

from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType

Alive   = BoolType('Alive')
Name    = StrType('Name')
Age     = IntType('Age')
Weight  = FloatType('Weight')

def test_metacall_derived_bool_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Alive(object())

def test_metacall_derived_str_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Name(object())

def test_metacall_derived_int_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Age(object())

def test_metacall_derived_float_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Weight(object())


class Alive(Bool):
    pass
class Name(Str):
    pass
class Age(Int):
    pass
class Weight(Float):
    pass

def test_class_derived_bool_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Alive(object())

def test_class_derived_str_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Name(object())

def test_class_derived_int_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Age(object())

def test_class_derived_float_type_doesnt_accept_general_objects():
    with pytest.raises(TypeError):
        Weight(object())

