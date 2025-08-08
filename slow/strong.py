#!/usr/bin/env python3

""" Strong typing. """

__all__ = [
    'to_vector',
    'to_array',
    'to_list_of_type',
    'is_jax_vector',
    'is_jax_array',
    'is_instance',
    'jax_array_type',
    'least_base_type',
]

import jax
import jaxlib
import jax.numpy as jnp

import types
import typing
from _collections_abc import dict_values

jax_array_type  = jaxlib.xla_extension.DeviceArray
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

def is_instance(obj, cls):

    """ Turducken typing. """

    if isinstance(cls, tuple):
        for scls in cls:
            if is_instance(obj, scls):
                return True
        else:
            return False

    if type(cls) is typing._UnionGenericAlias \
    and cls.__origin__ is typing.Union:
        return is_instance(obj, cls.__args__)

    if not isinstance(cls, types.GenericAlias):
        return isinstance(obj, cls)

    if not is_instance(obj, cls.__origin__):
        return False

    ocls = cls.__origin__
    args = cls.__args__

    if ocls is list:
        assert len(args) == 1
        itemcls = args[0]
        for item in obj:
            if not is_instance(item, itemcls):
                return False
        return True
    if ocls is dict:
        assert len(args) == 2
        keycls, valcls = args
        for key, val in obj.items():
            if not is_instance(key, keycls):
                return False
            if not is_instance(val, valcls):
                return False
        return True
    if ocls is tuple:
        for sobj, scls in zip(obj, args):
            if not is_instance(sobj, scls):
                return False
        return True

    raise TypeError(obj, cls)

to_thought = to_vector # compatability
__all__.append('to_thought')

