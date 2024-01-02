#!/usr/bin/env python3

__all__ = [
    'hybridmethod',
    'metamethod',
]

# random

import jax.random

class State:
    def __init__(self, key=42):
        self.key = jax.random.PRNGKey(key)
    def split(self, key):
        key, subkey = jax.random.split(key)
        return key, subkey
    def normal(self, shape):
        self.key, subkey = self.split(self.key)
        return jax.random.normal(subkey, shape)


# descriptors

class hybridmethod:

    """
        A descriptor that lets classes share code
        with their instances, ignoring similarly
        named methods in the metaclass.
    """

    def __init__(self, func):
        self.func = func

    def __get__(self, obj, cls):
        func = self.func
        if obj is not None:
            return func.__get__(obj)
        else:
            return func.__get__(cls)


class metamethod:

    """
        In python,
        the type of object is type,
        and the parent of type is object.

        This descriptor gives us the same behavior,
        so that types have the same method resolution
        order as instances when we call methods on them.
    """

    def __init__(self, func):
        self.func = func
        self.name = func.__name__

    def __get__(self, obj, cls):
        if obj is not None:
            return self.func.__get__(obj)
        meta = cls.__class__
        try:
            func = getattr(meta, self.name)
        except AttributeError:
            func = self.func
        finally:
            return func.__get__(cls)

