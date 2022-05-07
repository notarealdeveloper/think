#!/usr/bin/env python3

import pytest

from think import Object, Type
from think import Bool, Int, Str, Float
from think import BoolType, StrType, IntType, FloatType


def test_core_types_type_is_our_base_meta():
    assert type(Object) is Type
    assert type(Bool)   is Type
    assert type(Str)    is Type
    assert type(Int)    is Type
    assert type(Float)  is Type
