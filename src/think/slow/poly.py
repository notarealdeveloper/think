#!/usr/bin/env python3

from think import fast
from think import slow

__all__ = [
    'norm',
    'norm_l1',
    'norm_l2',
    'unit',
    'unit_l1',
    'unit_l2',
    'to_row',
    'to_col',
    'dot',
    'proj',
    'dist',
    'cos',
    'breed',
    'mean',
    'mix',
    'coordinates',
    'expand',
    'split',
    'project',
    'reject',
    'explained',
    'unexplained',
    'pre_attention_l1',
    'pre_attention_l2',
    'pre_attention_sm',
    'attention_l1',
    'attention_l2',
    'attention_sm',
    'dots',
    'setattr',
    'mixattr'
]

def norm(obj):
    t = slow.to_vector(obj)
    return fast.norm(t)

def norm_l1(obj):
    t = slow.to_vector(obj)
    return fast.norm_l1(t)

def norm_l2(obj):
    t = slow.to_vector(obj)
    return fast.norm_l2(t)

def unit(obj):
    t = slow.to_vector(obj)
    return fast.unit(t)

def unit_l1(obj):
    t = slow.to_vector(obj)
    return fast.unit_l1(t)

def unit_l2(obj):
    t = slow.to_vector(obj)
    return fast.unit_l2(t)

def to_row(obj):
    t = slow.to_vector(obj)
    return fast.to_row(t)

def to_col(obj):
    t = slow.to_vector(obj)
    return fast.to_col(t)

def dot(obj1, obj2):
    a = slow.to_vector(obj1)
    b = slow.to_vector(obj2)
    return fast.dot(a, b)

def proj(obj1, obj2):
    a = slow.to_vector(obj1)
    b = slow.to_vector(obj2)
    return fast.proj(a, b)

def dist(obj1, obj2):
    a = slow.to_vector(obj1)
    b = slow.to_vector(obj2)
    return fast.dist(a, b)

def cos(obj1, obj2):
    a = slow.to_vector(obj1)
    b = slow.to_vector(obj2)
    return fast.cos(a, b)

def breed(obj1, obj2):
    a = slow.to_vector(obj1)
    b = slow.to_vector(obj2)
    return fast.breed(a, b)

def mean(objs):
    ts = slow.to_array(objs)
    return fast.mean(ts)

def mix(objs):
    ts = slow.to_array(objs)
    return fast.mix(ts)

def coordinates(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.coordinates(ts, t)

def expand(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.expand(ts, t)

def split(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.split(ts, t)

def project(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.project(ts, t)

def reject(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.reject(ts, t)

def explained(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.explained(ts, t)

def unexplained(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.unexplained(ts, t)

def pre_attention_l1(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.pre_attention_l1(ts, t)

def pre_attention_l2(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.pre_attention_l2(ts, t)

def pre_attention_sm(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.pre_attention_sm(ts, t)

def attention_l1(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.attention_l1(ts, t)

def attention_l2(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.attention_l2(ts, t)

def attention_sm(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.attention_sm(ts, t)

def dots(objs, obj):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    return fast.dots(ts, t)

def setattr(objs, obj, value):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    v = slow.to_vector(value)
    return fast.setattr(ts, t, v)

def mixattr(objs, obj, value):
    ts = slow.to_array(objs)
    t = slow.to_vector(obj)
    v = slow.to_vector(value)
    return fast.mixattr(ts, t, v)
