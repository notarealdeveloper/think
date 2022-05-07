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
from jax.experimental import optimizers

import fast
import slow
logger = logging.getLogger(__name__)
from think.pretty import colors


class Knowledge:

    """ A bunch of feelings that calls itself knowledge. """

    def __init__(self, object):
        self.object = object
        self.compute()

    def compute(self, reset_wrong=True):

        object = self.object

        if not object.attrs:
            self.bits = []
            self.score = 1.0
            return self

        bits = []
        for n, (attr, value) in enumerate(object.attrs.items()):
            bit = {
                'attr': attr,
                'value': value,
                'feel': object.get(attr),
                'know': value.object,
            }
            bit['true'] = bit['feel'] == bit['know']

            if reset_wrong:
                if bit['true']:
                    bit['reset'] = False
                else:
                    object.setfeel(attr, value)
                    bit['reset'] = True

            bits.append(bit)

        correct = [bit for bit in bits if bit['true']]
        self.bits = bits
        self.score = len(correct)/len(bits)
        return self

    def pretty(self, sep=' '):
        if not self.bits:
            return f"No knowledge"
        else:
            return sep.join(self.__feelpretty(feeling) for feeling in self.bits)

    def __bool__(self):
        return bool(self.bits)

    def __len__(self):
        return len(self.bits)

    def __getitem__(self, item):
        return self.bits[item]

    def __iter__(self):
        return iter(self.bits)

    def __repr__(self):
        return self.pretty()

    def __feelpretty(self, feeling):
        # feel pretty (privately)
        from think.pretty import colors
        feel = feeling['feel']
        know = feeling['know']
        true = feeling['true']
        if true:
            return colors.green(feel)
        else:
            wrong = colors.red(feel)
            right = colors.blue(know)
            to    = colors.yellow('->')
            return f"({wrong} {to} {right})"


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
        A = slow.to_array(attr)    # 11 x T
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

    K = Knowledge(self)

    if K.score >= threshold:
        LOG(f"no training needed for {self!r}, knowledge already encoded {K.score:.2%}")
        return self

    LOG(f"training needed for {self!r}, knowledge encoded {K.score:.2%}, "
        f"will now train until {threshold:.2%}")

    opt_state = opt_init(t)
    loss = loss_fn(t, As, vs)

    while K.score < threshold:
        for n in range(steps_per_update):
            loss, grads = grad_fn_self(t, As, vs)
            opt_state = opt_update(0, grads, opt_state)
            t = get_params(opt_state)
        self.rethink(t)
        K = Knowledge(self)
        LOG(f"{self!r}: encoded {K.score:.2%} of knowledge, "
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


def encode(object):
    return encode_until_score(object)

def learn(cls):
    while not perfect(cls):
        for name, self in cls.instances().items():
            self.encode()
    print(colors.white(f"The system is now perfect ✨"))

def perfect(cls):
    for name, self in cls.instances().items():
        if Knowledge(self).score < 1.0:
            return False
    return True

def knowledge(cls):
    knowledge = {}
    for name, self in cls.instances().items():
        knowledge[self] = Knowledge(self)
    return knowledge

