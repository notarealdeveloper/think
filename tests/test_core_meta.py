#!/usr/bin/env python3

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


def test_core_meta_object_attribute():
    assert Type.object  is type
    assert BoolType.object  is type
    assert StrType.object   is type
    assert IntType.object   is type
    assert FloatType.object is type


def test_core_meta_base_attribute():
    assert BoolType.base  is Bool
    assert StrType.base   is Str
    assert IntType.base   is Int
    assert FloatType.base is Float


def test_core_meta_type_is_python_base_meta():
    assert type(Type)       is type
    assert type(BoolType)   is type
    assert type(StrType)    is type
    assert type(IntType)    is type
    assert type(FloatType)  is type


def test_core_meta_not_equal_implies_types_not_equal():
    assert Type('Ticker') != StrType('Ticker')


def test_core_meta_not_equal_implies_objects_not_equal():
    assert Type('Ticker')('SPY') != StrType('Ticker')('SPY')


if False:
    # obscure __classcell__ error occurring
    # when we make a class Char(Str): pass
    # *twice* (but not once), and this is
    # happening iff we cache types, so don't
    # cache types for now.
    def test_core_meta_idempotence():
        assert StrType('Name')   is StrType('Name')
        assert BoolType('Alive') is BoolType('Alive')
        assert IntType('Age')    is IntType('Age')
        assert FloatType('Size') is FloatType('Size')

