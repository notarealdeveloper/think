#!/usr/bin/env python3

from slow import cos
from think.core import Object, Str
from think import Char, Letter, Word, Sentence
from think import Digit, Digits
from think import Year, Month, Day
from think import Date
import think

import colors

think.thought_dim(1024)

# Words
def score(self, return_atoms=False, reset_wrong=False):
    if not self.attrs:
        return 1.0
    atoms = []
    total = 0
    correct = 0
    for n, (attr, value) in enumerate(self.attrs.items()):
        feel = self.get(attr)
        know = value.object
        if feel == know:
            atoms.append(colors.green(feel))
            correct += 1
        else:
            atoms.append(colors.red(feel))
            if reset_wrong:
                self.setfeel(attr, value)
                newfeel = self.getfeel(attr)
                reset_any = True
                print(f"Resetting {self}.{attr.name} = {value} (now {newfeel})")
        total += 1
    the_score = correct/total
    atoms = ' '.join(atoms)
    if return_atoms:
        return (the_score, atoms)
    else:
        return the_score

def score_and_atoms(self, reset_wrong=False):
    return score(self, return_atoms=True, reset_wrong=reset_wrong)

# Self training code
# TODO: put this in Object
def get_attrs_and_values(self):
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

def loss_fn(t, As, vs):
    feels = jnp.stack([fast.attention_l1(A, t) for A in As], axis=0)
    knows = jnp.stack(vs, axis=0)
    losses = (feels - knows)**2
    loss = losses.sum()
    return loss

from jax import vmap, value_and_grad
grad_fn_0   = value_and_grad(loss_fn)
grad_fn_0_2 = value_and_grad(loss_fn, argnums=[0,2])

from jax.experimental import optimizers

def encode_until_score(self, threshold=1.0, step_size=1e-2, steps_per_update=100):

    # no need to do anything for no-knowledge objects
    if not self.attrs:
        return self

    # don't yammer about contextual types
    PRINT = print if type(self).primary else lambda x: None

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    (self, t), (attrs, As), (values, vs) = get_attrs_and_values(self)

    s, atoms = score_and_atoms(self)
    if s >= threshold:
        PRINT(f"no training needed for {self!r}, knowledge already encoded {s:.2%}")
        return self

    PRINT(atoms)
    PRINT(f"training needed for {self!r}, knowledge encoded {s:.2%}, "
          f"will now train until {threshold:.2%}")

    opt_state = opt_init(t)
    loss = loss_fn(t, As, vs)

    while s < threshold:
        for n in range(steps_per_update):
            loss, grads = grad_fn_0(t, As, vs)
            opt_state = opt_update(0, grads, opt_state)
            t = get_params(opt_state)
        self.rethink(t)
        #for value, v in zip(values, vs):
        #    value.rethink(v)
        s, atoms = score_and_atoms(self)
        PRINT(f"{self!r}: encoded knowledge {s:.2%}, desired {threshold:.2%} (loss: {loss})")
    return self


def encode_until_loss(self, threshold=1e-1, step_size=1e-2):

    if not self.attrs:
        # no need to do anything for no-knowledge objects
        return self

    # don't yammer about contextual types
    PRINT = print if type(self).primary else lambda x: None

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    (self, t), (attrs, As), (values, vs) = get_attrs_and_values(self)

    loss = loss_fn(t, As, vs)
    if loss <= threshold:
        PRINT(f"no training needed for {self!r}, loss is already {loss}")
        return self

    PRINT(f"training needed for {self!r}, loss is {loss}, will now train until {threshold}")

    opt_state = opt_init(t)
    while loss > threshold:
        loss, grads = grad_fn_0(t, As, vs)
        opt_state = opt_update(0, grads, opt_state)
        t = get_params(opt_state)
        PRINT(f"{self!r}: encoded knowledge to loss: {loss}")
    self.rethink(t)
    #for value, v in zip(values, vs):
    #    value.rethink(v)
    return self

#encode_knowledge = encode_until_loss
encode_knowledge = encode_until_score

# End of self training code

# Here's an example:
a = Word('supercalifragilisticexpialidocious')
b = Word('antidisestablishementarianism')
c = Word('cupcake')
d = Word('hotdog')
for w in ('hey', 'there', 'babycakes', 'how', 'are', 'you', 'doing'):
    Word(w)

s = Sentence("This is a fucking brain")
t = Sentence("How cool is this")
u = Sentence("That bitch crazy")
v = Sentence("Yo mama so fat when she sits around the house she really sits around the house")

def perfect():
    for name, object in Object.instances().items():
        if score(object) < 1.0:
            return False
    return True

while not perfect():
    for name, object in Object.instances().items():
        encode_knowledge(object)

print(f"The system is perfect.")

for word in (a,b,c,d):
    for n, letter in enumerate(word.object):
        assert word.get(Letter[n]) == letter

for sentence in (s,t,u,v):
    for n, word in enumerate(sentence.object):
        assert sentence.get(Word[n]) == word

for sentence in (s,t,u,v):
    for n, word in enumerate(sentence.object):
        assert sentence.get(Word[n]) == word
        word = Word(word)
        for m, letter in enumerate(word.object):
            assert word.get(Letter[m]) == letter

