#!/usr/bin/env python3

import pytest

from think import Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


def test_metaclass_idempotence():
    assert StrType('Name') is StrType('Name')
    assert BoolType('Alive') is BoolType('Alive')
    assert IntType('Age') is IntType('Age')
    assert FloatType('Weight') is FloatType('Weight')


def test_metaclass_type_attribute():
    assert BoolType.type is type
    assert StrType.type is type
    assert IntType.type is type
    assert FloatType.type is type


def test_metaclass_base_attribute():
    assert BoolType.base is Bool
    assert StrType.base is Str
    assert IntType.base is Int
    assert FloatType.base is Float


def test_metaclass_not_equal_causes_instances_not_equal():

    class StrType1(StrType):
        pass
    class StrType2(StrType):
        pass

    SPY1 = StrType1('Ticker')('SPY')
    SPY2 = StrType2('Ticker')('SPY')

    assert SPY1 != SPY2


def test_metaclass_not_equal_causes_classes_not_equal():

    class StrType1(StrType):
        pass
    class StrType2(StrType):
        pass

    assert StrType1('Ticker') != StrType2('Ticker')


def test_class_not_equal_causes_instances_not_equal():

    SPY1 = StrType('Ticker1')('SPY')
    SPY2 = StrType('Ticker2')('SPY')

    assert SPY1 != SPY2


def test_that_base_can_be_passed_in_metaclass_body():

    class Ticker(Str):
        pass

    class TickerType(Type):
        base = Ticker

    ETF = TickerType('ETF')

    assert ETF.base is Ticker


def test_that_base_can_be_passed_in_metaclass_new():

    class StrType(Type):
        def __new__(cls, name):
            return Type.__new__(cls, name, Str)

    Name = StrType('Name')

    assert Name.base is Str



