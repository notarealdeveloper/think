#!/usr/bin/env python3

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


# derived meta
class Boolean(BoolType): pass
class String(StrType): pass
class Integer(IntType): pass
class Floating(FloatType): pass
class StrType1(StrType): pass
class StrType2(StrType): pass


def test_derived_meta_object_attribute():
    assert Boolean.object  is type
    assert String.object   is type
    assert Integer.object  is type
    assert Floating.object is type


def test_derived_meta_base_attribute():
    assert Boolean.base  is Bool
    assert String.base   is Str
    assert Integer.base  is Int
    assert Floating.base is Float


def test_derived_meta_not_equal_implies_types_not_equal():
    assert StrType1('Ticker') != StrType2('Ticker')


def test_derived_meta_not_equal_implies_objects_not_equal():
    assert StrType1('Ticker')('SPY') != StrType2('Ticker')('SPY')



# ways of requesting the bases that our new class will subclass

class Ticker(Str):
    pass

class TickerType1(Type):
    base = Ticker

class TickerType2(Type):
    def __new__(cls, name):
        return Type.__new__(cls, name, Ticker)

def test_derived_meta_returns_subclasses_of_base_if_passed_in_metaclass_body():
    ETF = TickerType1('ETF')
    assert issubclass(ETF, Ticker)


def test_derived_meta_returns_subclasses_of_base_if_passed_in_metaclass_new():
    ETF = TickerType2('ETF')
    assert issubclass(ETF, Ticker)


if False:
    # obscure __classcell__ error occurring
    # when we make a class Char(Str): pass
    # *twice* (but not once), and this is
    # happening iff we cache types, so don't
    # cache types for now.
    def test_idempotence_for_derived_meta():
        assert Boolean('Alive') == Boolean('Alive')
        assert String('Name')   == String('Name')
        assert Integer('Age')   == Integer('Age')
        assert Floating('Size') == Floating('Size')


