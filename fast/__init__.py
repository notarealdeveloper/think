#!/usr/bin/env python3

"""
    Thought geometry: JIT functions
"""

__all__ = [
    # unary functions on vectors
    'norm',
    'norm_l1',
    'norm_l2',
    'unit',
    'unit_l1',
    'unit_l2',
    'to_row',
    'to_col',

    # binary functions on vectors
    'dot',
    'proj',
    'dist',
    'cos',
    'breed',

    # unary functions of a hard coded plural argument
    'mean',
    'mix',

    # binary functions with at least one hard coded plural argument
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

    # ternary functions with at least one hard coded plural argument
    'setattr',
    'mixattr',
]

import jax
import jax.numpy as jnp
from jax import jit
from jax.nn import softmax

##################################
### UNARY FUNCTIONS OF VECTORS ###
##################################

@jit
def norm(t):
    return jnp.sqrt(dot(t, t))

@jit
def norm_l1(t):
    return jnp.sum(t)

@jit
def norm_l2(t):
    return jnp.sqrt(dot(t, t))

@jit
def unit(t):
    return t/norm(t)

@jit
def unit_l1(t):
    return t/norm_l1(t)

@jit
def unit_l2(t):
    return t/norm_l2(t)

@jit
def to_row(t):
    assert t.ndim == 1
    return jnp.expand_dims(t, axis=0)

@jit
def to_col(t):
    assert t.ndim == 1
    return jnp.expand_dims(t, axis=-1)

###################################
### BINARY FUNCTIONS OF VECTORS ###
###################################

@jit
def dot(a, b):
    return (a*b).sum()

@jit
def proj(a, b):
    u = unit(b)
    return dot(a, u)*u

@jit
def rej(a, b):
    return a - proj(a, b)

@jit
def dist(a, b):
    return norm(a - b)

@jit
def cos(a, b):
    return dot(unit(a), unit(b))

@jit
def breed(a, b):
    return jnp.add(a, b)/jnp.sqrt(2)


#######################################################
### UNARY FUNCTIONS OF A HARD CODED PLURAL ARGUMENT ###
#######################################################

@jit
def mean(ts):
    return jnp.mean(ts, axis=0)

@jit
def mix(ts):
    n = ts.shape[0]
    return jnp.sum(ts, axis=0)/jnp.sqrt(n)


#####################################################################
### BINARY FUNCTIONS WITH AT LEAST ONE HARD CODED PLURAL ARGUMENT ###
#####################################################################

@jit
def coordinates(ts, t):
    coefs, err, rank, svals = jnp.linalg.lstsq(ts.T, to_col(t))
    return coefs

@jit
def expand(ts, t):
    ws = coordinates(ts, t)
    return ws*ts

@jit
def split(ts, t):
    basis = expand(ts, t)
    inner = jnp.sum(basis, axis=0)
    outer = t - inner
    return jnp.stack([inner, outer], axis=0)

@jit
def project(ts, t):
    basis = expand(ts, t)
    inner = jnp.sum(basis, axis=0)
    return inner

@jit
def reject(ts, t):
    return t - project(ts, t)

@jit
def explained(ts, t):
    i = project(ts, t)
    return (norm(i)/norm(t))**2

@jit
def unexplained(ts, t):
    o = reject(ts, t)
    return (norm(o)/norm(t))**2

@jit
def pre_attention_l1(ts, t):
    q = project(ts, t)
    dots = jnp.dot(ts, q)
    sims = dots/norm_l1(dots)
    return sims

@jit
def pre_attention_l2(ts, t):
    q = project(ts, t)
    dots = jnp.dot(ts, q)
    sims = dots/norm_l2(dots)
    return sims

@jit
def pre_attention_sm(ts, t):
    q = project(ts, t)
    dots = jnp.dot(ts, q)
    sims = softmax(dots)
    return sims

@jit
def attention_l1(ts, t):
    sims = pre_attention_l1(ts, t)
    return sims @ ts

@jit
def attention_l2(ts, t):
    sims = pre_attention_l2(ts, t)
    return sims @ ts

@jit
def attention_sm(ts, t):
    sims = pre_attention_sm(ts, t)
    return sims @ ts

@jit
def dots(ts, t):
    return jnp.dot(ts, t)

######################################################################
### TERNARY FUNCTIONS WITH AT LEAST ONE HARD CODED PLURAL ARGUMENT ###
######################################################################

@jit
def setattr(ts, t, v):
    t_in, t_out = split(ts, t)
    v_in, v_out = split(ts, v)
    return v_in + t_out

@jit
def mixattr(ts, t, v):
    t_in, t_out = split(ts, t)
    v_in, v_out = split(ts, v)
    v_frac = (norm(v_in)/norm(v))**2
    t_frac = (norm(t_in)/norm(t))**2
    weight = (1 - t_frac)*(v_frac)
    t_in2 = (weight)*v_in + (1 - weight)*t_in
    return t_in2 + t_out

