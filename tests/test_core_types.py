#!/usr/bin/env python3

import pytest

from think import Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


def test_that_base_can_be_passed_in_metaclass_body():

    class Ticker(Str):
        type = str

    class TickerType(Type):
        base = Ticker

    ETF = TickerType('ETF')

    assert issubclass(ETF, Ticker)


def test_that_bases_can_be_passed_in_metaclass_new():

    class StrType(Type):
        def __new__(cls, name):
            return Type.__new__(cls, name, Str)

    Name = StrType('Name')

    assert issubclass(Name, Str)
