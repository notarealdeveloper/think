#!/usr/bin/env python3

import pytest

from think import Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


def test_core_meta_idempotence():
    assert StrType('Name')   is StrType('Name')
    assert BoolType('Alive') is BoolType('Alive')
    assert IntType('Age')    is IntType('Age')
    assert FloatType('Size') is FloatType('Size')



def test_core_meta_attribute_dot_base():
    assert BoolType.base    is Bool
    assert StrType.base     is Str
    assert IntType.base     is Int
    assert FloatType.base   is Float


def test_class_not_equal_causes_instances_not_equal():
    SPY1 = StrType('Ticker1')('SPY')
    SPY2 = StrType('Ticker2')('SPY')
    assert SPY1 != SPY2

