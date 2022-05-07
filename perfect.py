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


def get_data_for_self_training(self, only_train_wrong=False):
    As = []
    vs = []
    attrs = []
    values = []
    for attr, value in self.attrs.items():
        feel = self.get(attr)
        know = value.object
        if only_train_wrong and feel == know:
            continue
        attrs.append(attr)
        values.append(value)
        A = slow.to_array(attr)
        v = value.think()
        vs.append(v)
        As.append(A)
    t = self.think()
    return (self, t), (attrs, As), (values, vs)


def learn_until_score(self, threshold=1.0, step_size=1e-2,
                      optimizer=None, steps_per_update=20,
                      only_train_wrong=True):

    # don't yammer about contextual types
    LOG = print # if type(self).primary else logger.debug

    score = self.score()
    if score >= threshold:
        LOG(f"no training needed for {self!r}, knowledge already encoded {score:.2%}")
        return self

    loop = 0
    num_proj = 0
    num_grad = 0

    while True:
        # train with projections
        self.reset_wrong()
        score = self.score()
        num_proj += 1

        if score >= threshold:
            LOG(f"{self!r}: projection got it: (projs={num_proj} grads={num_grad})")
            return self

        # train with gradients
        if loop == 0:
            LOG(f"training needed for {self!r}, knowledge encoded {score:.2%}, "
                f"will now train until {threshold:.2%}")
            (self, t), (attrs, As), (values, vs) = \
                get_data_for_self_training(self, only_train_wrong)
            if optimizer is None:
                optimizer = optimizers.adam
            opt_init, opt_update, get_params = optimizer(step_size)
            opt_state = opt_init(t)

        for n in range(steps_per_update):
            loss, grads = grad_fn_self(t, As, vs)
            opt_state = opt_update(0, grads, opt_state)
            t = get_params(opt_state)
        num_grad += 1
        self.rethink(t)

        score = self.score()
        if score >= threshold:
            LOG(f"{self!r}: gradients got it: (projs={num_proj} grads={num_grad})")
            # note: these are still naive projections
            return self

        LOG(f"{self!r}: end of loop {loop}. {score:.2%} of knowledge encoded, "
            f"desired {threshold:.2%} (loss={loss}, projs={num_proj} grads={num_grad})")

        loop += 1
    return self


def learn_until_loss(self, threshold=1e-1, step_size=1e-3, optimizer=None,
                    steps_per_update=20):

    # no need to do anything for no-knowledge objects
    if not self.attrs:
        return self

    if optimizer is None:
        optimizer = optimizers.adam

    # don't yammer about contextual types
    LOG = print if type(self).primary else logger.debug

    opt_init, opt_update, get_params = optimizer(step_size)

    (self, t), (attrs, As), (values, vs) = get_data_for_self_training(self, only_train_wrong)

    loss = loss_fn(t, As, vs)
    if loss <= threshold:
        LOG(f"no training needed for {self!r}, loss is already {loss}")
        return self

    LOG(f"training needed for {self!r}, loss is {loss}, will now train until {threshold}")

    opt_state = opt_init(t)
    while loss > threshold:
        for n in range(steps_per_update):
            loss, grads = grad_fn_self(t, As, vs)
            opt_state = opt_update(0, grads, opt_state)
            t = get_params(opt_state)
        self.rethink(t)
        LOG(f"{self!r}: encoded knowledge to loss: {loss}")
    return self


