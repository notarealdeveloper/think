#!/usr/bin/env python

"""
    Gradient based learning.
"""

__all__ = [
]

import logging
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.example_libraries import optimizers

from think import fast
from think import slow
logger = logging.getLogger(__name__)

MIN_LOSS = 1e-6

##########################
### SELF TRAINING CODE ###
##########################

def loss_fn(t, As, vs):
    # As/vs may be Python lists; vmap expects an array-like.
    # Each A is (num_instances_for_attr, thought_dim). We stack into
    # (num_attrs, num_instances_for_attr, thought_dim).
    As = jnp.stack(As, axis=0)
    knows = jnp.stack(vs, axis=0)
    feels = vmap(lambda A: slow.attention_l1(A, t))(As)
    losses = (feels - knows)**2
    loss = losses.sum()
    return loss


grad_fn_self = jit(value_and_grad(loss_fn))
grad_fn_all  = value_and_grad(loss_fn, argnums=[0,1,2])


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
    return (attrs, As), (values, vs)


def learn_until_score(self, threshold=1.0, step_size=1e-2,
                      optimizer=None, steps_per_update=20,
                      only_train_wrong=True, max_steps=float('inf')):

    # don't yammer about contextual types
    LOG = print # if type(self).primary else logger.debug

    score = self.score()
    if score >= threshold:
        LOG(f"no training needed for {self!r}, knowledge already encoded {score:.2%}")
        return self

    steps = 0
    num_proj = 0
    num_grad = 0

    while steps < max_steps:
        # train with projections
        self.reset_wrong()
        t = self.think()
        
        (attrs, As), (values, vs) = get_data_for_self_training(self, only_train_wrong)
        if not As:
            LOG(f"{self!r}: nothing left to train")
            return self
        
        score = self.score()
        num_proj += 1

        if score >= threshold:
            LOG(f"{self!r}: projection got it: (projs={num_proj} grads={num_grad})")
            return self

        # train with gradients
        if steps == 0:
            LOG(f"training needed for {self!r}, knowledge encoded {score:.2%}, "
                f"will now train until {threshold:.2%}")
            if optimizer is None:
                optimizer = optimizers.adam
            opt_init, opt_update, get_params = optimizer(step_size)
            opt_state = opt_init(t)

        for n in range(steps_per_update):
            loss, grads = grad_fn_self(t, As, vs)
            if loss < MIN_LOSS:
                break
            opt_state = opt_update(steps, grads, opt_state)
            t = get_params(opt_state)
            steps += 1
        num_grad += 1
        self.rethink(t)

        score = self.score()
        if score >= threshold:
            LOG(f"{self!r}: gradients got it: (projs={num_proj} grads={num_grad})")
            # note: these are still naive projections
            return self

        LOG(f"{self!r}: finished {steps} steps. {score:.2%} of knowledge encoded, "
            f"desired {threshold:.2%} (loss={loss}, projs={num_proj} grads={num_grad})")

    LOG(f"{self!r}: reached max steps, moving on.")
    return self


def learn_until_loss(self, threshold, step_size=1e-3,
                     optimizer=None, steps_per_update=20,
                     only_train_wrong=False, max_steps=float('inf')):

    # no need to do anything for no-knowledge objects
    if not self.attrs:
        return self

    if optimizer is None:
        optimizer = optimizers.adam

    # don't yammer about contextual types
    LOG = print if type(self).primary else logger.debug

    opt_init, opt_update, get_params = optimizer(step_size)

    t = self.think()
    (attrs, As), (values, vs) = get_data_for_self_training(self, only_train_wrong)

    loss = loss_fn(t, As, vs)
    if loss <= threshold:
        LOG(f"no training needed for {self!r}, loss is already {loss}")
        return self

    LOG(f"training needed for {self!r}, loss is {loss}, will now train until {threshold}")

    opt_state = opt_init(t)
    steps = 0
    while steps < max_steps:
        for n in range(steps_per_update):
            loss, grads = grad_fn_self(t, As, vs)
            if loss < MIN_LOSS:
                break
            opt_state = opt_update(steps, grads, opt_state)
            t = get_params(opt_state)
            steps += 1
            if loss < threshold:
                LOG(f"{self!r}: training finished, encoded knowledge to loss: {loss}")
        self.rethink(t)
        self.reset_wrong()
        LOG(f"{self!r}: encoded knowledge to loss: {loss}")
    LOG(f"{self!r}: reached max steps, moving on.")
    return self

