#!/usr/bin/env python3

"""
    Take a brain and make it perfect.
"""

__all__ = [
    # this file contains gradient based learning,
    # and its internals should be segregated from
    # the rest of the system at all costs.
    #
    # anything you want from this module,
    # you pull it out explicitly.
]

import logging
import jax.numpy as jnp
from jax import vmap, value_and_grad
try:
    # jax optimizers are here in newer versions
    from jax.example_libraries import optimizers
except:
    # jax optimizers are here in older versions
    from jax.experimental import optimizers

import fast
import slow
logger = logging.getLogger(__name__)
from think.pretty import colors



##########################
### SELF TRAINING CODE ###
##########################

def loss_fn(t, As, vs):
    feels = jnp.stack([fast.attention_l1(A, t) for A in As], axis=0)
    knows = jnp.stack(vs, axis=0)
    losses = (feels - knows)**2
    loss = losses.sum()
    return loss


grad_fn_self = value_and_grad(loss_fn)
grad_fn_meta = value_and_grad(loss_fn, argnums=[0,1,2])


def get_data_for_self_training(self):
    attrs_and_values = list(self.attrs.items())
    As = []
    vs = []
    attrs = []
    values = []
    for attr, value in attrs_and_values:
        attrs.append(attr)
        values.append(value)
        A = slow.to_array(attr)
        v = value.think()
        vs.append(v)
        As.append(A)
    t = self.think()
    return (self, t), (attrs, As), (values, vs)


def encode_until_score(self, threshold=1.0, step_size=1e-2,
                        optimizer=None, steps_per_update=100):

    # no need to do anything for no-knowledge objects
    if not self.attrs:
        return self

    if optimizer is None:
        optimizer = optimizers.adam

    # don't yammer about contextual types
    LOG = print if type(self).primary else logger.debug

    opt_init, opt_update, get_params = optimizer(step_size)

    (self, t), (attrs, As), (values, vs) = get_data_for_self_training(self)

    self.reset_wrong()
    score = self.score()

    if score >= threshold:
        LOG(f"no training needed for {self!r}, knowledge already encoded {score:.2%}")
        return self

    LOG(f"training needed for {self!r}, knowledge encoded {score:.2%}, "
        f"will now train until {threshold:.2%}")

    loss = loss_fn(t, As, vs)
    opt_state = opt_init(t)

    while score < threshold:
        for n in range(steps_per_update):
            loss, grads = grad_fn_self(t, As, vs)
            opt_state = opt_update(0, grads, opt_state)
            t = get_params(opt_state)
        self.rethink(t)
        self.reset_wrong()
        score = self.score()
        LOG(f"{self!r}: encoded {score:.2%} of knowledge, "
            f"desired {threshold:.2%} (loss: {loss})")
    return self


def encode_until_loss(self, threshold=1e-1, step_size=1e-2, optimizer=None):

    # no need to do anything for no-knowledge objects
    if not self.attrs:
        return self

    if optimizer is None:
        optimizer = optimizers.adam

    # don't yammer about contextual types
    LOG = print if type(self).primary else logger.debug

    opt_init, opt_update, get_params = optimizer(step_size)

    (self, t), (attrs, As), (values, vs) = get_data_for_self_training(self)

    loss = loss_fn(t, As, vs)
    if loss <= threshold:
        LOG(f"no training needed for {self!r}, loss is already {loss}")
        return self

    LOG(f"training needed for {self!r}, loss is {loss}, will now train until {threshold}")

    opt_state = opt_init(t)
    while loss > threshold:
        loss, grads = grad_fn_self(t, As, vs)
        opt_state = opt_update(0, grads, opt_state)
        t = get_params(opt_state)
        LOG(f"{self!r}: encoded knowledge to loss: {loss}")
    self.rethink(t)
    return self


def learn(cls):
    while not perfect(cls):
        for name, self in cls.instances().items():
            self.encode_until_score(threshold=1.0)
    print(colors.white(f"The system is now perfect ✨"))

def perfect(cls):
    for name, self in cls.instances().items():
        if self.score() < 1.0:
            return False
    return True

