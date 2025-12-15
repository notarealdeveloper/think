#!/usr/bin/env python

from think import Object, Type
from think import Bool, Int, Str, Float

def test_core_types_type_is_our_base_meta():
    assert type(Object) is Type
    assert type(Bool)   is Type
    assert type(Str)    is Type
    assert type(Int)    is Type
    assert type(Float)  is Type

