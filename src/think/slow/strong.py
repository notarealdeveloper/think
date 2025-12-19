#!/usr/bin/env python

""" Strong typing. """

__all__ = [
    'to_vector',
    'to_array',
    'to_thought',
    'to_list_of_type',
    'is_jax_vector',
    'is_jax_array',
    'jax_array_type',
    'least_base_type',
]

import jax
import jaxlib
import jax.numpy as jnp

import types
import typing
import is_instance
from _collections_abc import dict_values

jax_array_type = jax.Array
jax_tracer_type = jax.core.Tracer

def to_vector(arg):
    if isinstance(arg, jax_tracer_type):
        return arg # let jax trace
    if isinstance(arg, jax_array_type):
        assert arg.ndim == 1
        return arg
    if hasattr(arg, '__array__'):
        arg = arg.__array__()
        assert arg.ndim == 1
        return arg
    raise TypeError(f"Cannot coerce to vector: {arg!r}")


def to_array(arg):
    if isinstance(arg, jax_tracer_type):
        return arg # let jax trace
    if isinstance(arg, jax_array_type):
        assert arg.ndim == 2
        return arg
    if isinstance(arg, dict):
        arg = list(arg.values())
    if isinstance(arg, dict_values):
        arg = list(arg)
    if isinstance(arg, (list, tuple, set)):
        vectors = [to_vector(o) for o in arg]
        return jnp.stack(vectors, axis=0)
    if hasattr(arg, '__array__'):
        arg = arg.__array__()
        assert arg.ndim == 2
        return arg
    raise TypeError(f"Cannot coerce to array: {arg!r}")


def to_list_of_type(arg, type):
    if isinstance(arg, type):
        return [arg]
    elif is_instance(arg, list[type]):
        return arg
    elif is_instance(arg, tuple[type]):
        return list(arg)
    else:
        clsname = type.__name__
        raise TypeError(f"Cannot coerce to list of {clsname}: {arg!r}")

def is_jax_array(arg):
    return isinstance(arg, jax_array_type)

def is_jax_vector(arg):
    return isinstance(arg, jax_array_type) and arg.ndim == 1

def least_base_type(*types):
    from functools import reduce
    from operator import and_
    from collections import Counter
    return next(iter(reduce(and_, (Counter(t.mro()) for t in types))))

to_thought = to_vector # compatability

