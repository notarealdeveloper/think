#!/usr/bin/env python3

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


# derived types
class Name(Str):
    pass

class Ticker(Object):
    type = str


# derived metas
class StrType1(StrType):
    pass

class StrType2(StrType):
    pass

class TickerType(Type):
    base = Ticker


def test_derived_meta_not_equal_implies_types_not_equal():
    assert StrType1('Ticker') != StrType2('Ticker')


def test_derived_meta_not_equal_implies_objects_not_equal():
    SPY1 = StrType1('Ticker')('SPY')
    SPY2 = StrType2('Ticker')('SPY')
    assert SPY1 != SPY2


def test_derived_meta_base_can_be_passed_in_metaclass_body():
    ETF = TickerType('ETF')
    assert issubclass(ETF, Ticker)


