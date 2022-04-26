#!/usr/bin/env python3

import abc
import weakref
from functools import partial

import jax
import jax.random
import jax.numpy as jnp
from jax import jit
from jax import curry
from jax import grad, value_and_grad
from jax import tree_util
from jax.tree_util import register_pytree_node, register_pytree_node_class
from jax.tree_util import Partial

import fast
import slow


# random.py

THOUGHT_DIM = 1024

class State:
    def __init__(self, key=42):
        self.key = jax.random.PRNGKey(key)
    def split(self, key):
        key, subkey = jax.random.split(key)
        return key, subkey
    def normal(self, shape):
        self.key, subkey = self.split(self.key)
        return jax.random.normal(subkey, shape)

STATE = State()


# thoughts.py

def new_thought():
    denominator = jnp.sqrt(THOUGHT_DIM)
    return STATE.normal([THOUGHT_DIM])/denominator


class Thought:

    def __init__(self, t=None):
        self._t = t if t is not None else new_thought()
        self.__class__.instances.add(self)

    def think(self):
        return self._t

    def rethink(self, t):
        self._t = t

    def deps(self):
        return {'_t': self._t}

    def __repr__(self):
        return f"{self.__class__.__name__}({self._t})"

    # for tracking memory usage
    instances = weakref.WeakSet()

    @classmethod
    def gigabytes(cls):
        bytes = 4*THOUGHT_DIM*len(cls.instances)
        return bytes/(1024**3)

    @classmethod
    def active(cls):
        return len(cls.instances)


# types.py

class Object:

    instances = weakref.WeakSet()

    def __init__(self, object, thought=None):
        self.thought = Thought() if thought is None else thought
        self.object = object
        self.attrs = {}
        self.knowledge_generators = {}
        self.__class__.instances.add(self)

    def think(self):
        t = self.thought.think()
        t = self.apply_knowledge(t)
        return t

    def apply_knowledge(self, t):
        for attr, value in self.attrs.items():
            t = slow.hardset(attr, t, value)
        return t

    def rethink(self, ts):
        params = self.params()
        assert len(params) == len(ts)
        for param, t in zip(params, ts):
            if not isinstance(param, Thought):
                print(f"rethink encountered problem (breakpoint 1)")
                breakpoint()
            if not (hasattr(t, 'shape') and t.shape == (THOUGHT_DIM,)): # is_thought
                print(f"rethink encountered problem (breakpoint 2)")
                breakpoint()
            old = param.think()
            new = t
            param.rethink(new)
            print(f"{self} rethinking parameter, delta is:", fast.dist(old, new))

        return self.thought.rethink(t)

    def deps(self):
        return {'thought': self.thought}

    def params(self):
        return [self.thought]

    def set(self, attr, value):
        """ This should not call hardset!
            We need to keep the errors in our feelings around
            so the system that produced them can train itself.
        """
        if not isinstance(value, attr.object):
            value = attr(value)
        self.attrs[attr] = value
        self.remember(attr, value)
        return self

    def get(self, attr, hard_or_soft='soft', feel_or_know='both'):
        if not isinstance(attr, Type):
            raise TypeError(f"get: attr {attr} is not an instance of Type")
        if attr not in self.attrs:
            feel = attr.project(self)
            print(f"get: {self} asked for {Attr}, but no explicit knowledge exists. "
                  f"returning feeling: {feel}")
            return feel

        value = self.attrs[attr]

        self.remember(attr, value)

        # now just give an answer
        # we'll learn more from the introspection in the background ;)
        feel = attr.project(self)
        know = attr.project(value)

        # handle feeling vs knowing
        if feel_or_know == 'feel':
            thought = feel
        elif feel_or_know == 'know':
            thought = know
        elif feel_or_know == 'both':
            thought = slow.mix([feel, know])
        else:
            raise ValueError(f"get: bad value for feel_or_know: {feel_or_know!r}")

        if hard_or_soft == 'soft':
            return thought
        elif hard_or_soft == 'hard':
            return attr.invert(thought)
        else:
            raise ValueError(f"get: bad value for hard_or_soft: {hard_or_soft!r}")

    def hardget(self, attr, feel_or_know='both'):
        return self.get(attr, hard_or_soft='hard', feel_or_know='both')

    def remember(self, attr, value):
        if (attr, value) not in self.knowledge_generators:
            gen = self.knowledge_generator(attr, value)
            self.knowledge_generators[(self, attr, value)] = gen

    def knowledge_generator(self, attr, value):
        """ Using differences between what we feel and what we know
            to train the entire system that produced both. """
        objects = [self, attr, value]
        project = Partial(attr.project)
        knowledge_generator = make_train_loop(objects, ugly_diffget, project=project)
        sentence = f"{self.object}.{attr} = {value.object}"
        print(f"new knowledge generator for sentence {sentence!r}")
        return knowledge_generator

    def learn(self, steps=100):
        for (self, attr, value), knowledge_generator in self.knowledge_generators.items():
            for step in range(steps):
                allparams = next(knowledge_generator)

            objects = (self, attr, value)
            for params, obj in zip(allparams, objects):
                obj.rethink(params)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.object})"


class Int(Object):
    pass

class Float(Object):
    pass

class Str(Object):
    pass

class Bool(Object):
    pass


class Type(Object):

    def __init__(self, name, base):
        self.name = name
        self.base = base
        self.object = type(name, (base,), {})
        self.thought = Thought()
        self.attrs = {}
        self.knowledge_generators = {}

    def __call__(self, *args, **kwds):
        return self.object(*args, **kwds)

    def __repr__(self):
        return f"{self.name}"

    def think(self, object=None):
        """ Types can think about themselves, or about other objects. """
        if object is None:
            return self.thought.think()
        return self.think_about(object)

    def think_about(self, object):
        return slow.mix([self.thought, self.project(object)])

    @abc.abstractmethod
    def project(self, object, params=None):
        raise NotImplementedError

    @abc.abstractmethod
    def invert(self, object, params=None):
        raise NotImplementedError

    @abc.abstractmethod
    def params(self):
        raise NotImplementedError

    def __array__(self):
        return slow.to_array(self.deps())


class Integer(Type):
    def __init__(self, name, base=Int):
        super().__init__(name, base)
        self.b = self.object.thought
        self.w = Thought()

    def __call__(self, s):
        return self.object(s)

    def deps(self):
        return {'b': self.b, 'w': self.w}

    def params(self):
        return [self.b, self.w]

    def invert(self, object):
        params = params or self.deps()
        t = slow.to_thought(object)
        b = params['b']
        w = params['w']
        coords = slow.coordinates({'w': params['w']}, t-b)
        return coords['w']

    def project(self, object):
        s = self.unwrap(object)
        return self.ugly_project(self.deps(), s)

    @staticmethod
    def ugly_project(params, s):
        b, w = params
        return b.think() + s*w.think()


class String(Type):

    def __init__(self, name, base=Str):
        super().__init__(name, base)
        self.memory = {}

    def __call__(self, str):
        if str in self.memory:
            return self.memory[str]
        o = self.object(str)
        self.memory[str] = o
        return o

    def deps(self):
        return {k:v.thought for k,v in self.memory.items()}

    def params(self):
        return [v.thought for v in self.memory.values()]

    def invert(self, object):
        t       = slow.to_thought(object)
        sims    = slow.pre_attention_l1(self.params(), object)
        idx     = int(jnp.argmax(sims))
        key     = list(params.keys())[idx]
        return key

    def project(self, object, params=None):
        params = params if params is not None else self.params()
        return slow.attention_l1(params, object)


class Boolean(Type):

    def __init__(self, name, base=Bool):
        super().__init__(name, base)
        self.instances = {bool:self.object(bool) for bool in (True, False)}

    def __call__(self, bool):
        return self.instances[bool]

    def deps(self):
        return {k:v.thought for k,v in self.instances.items()}

    def params(self):
        return [v.thought for v in self.instances.values()]

    def invert(self, object, params=None):
        params  = params or self.deps()
        sims    = slow.pre_attention_l1(params, object)
        idx     = int(jnp.argmax(sims))
        key     = list(params.keys())[idx]
        return key

    def project(self, object, params=None):
        params = params if params is not None else self.params()
        return slow.attention_l1(params, object)


########################################
###             ugly.py              ###
### bottom-up machine learning code  ###
########################################


def make_loss(func):
    def loss_fn(*args, **kwds):
        return jnp.sum(func(*args, **kwds))
    return loss_fn

def tree_think(o):
    if isinstance(o, dict):
        return {k:tree_think(v) for k,v in o.items()}
    if isinstance(o, list):
        return [tree_think(v) for v in o]
    if isinstance(o, tuple):
        return tuple([tree_think(v) for v in o])
    if isinstance(o, Type):
        return o.__array__()
    if isinstance(o, Object):
        return o.think()
    if isinstance(o, Thought):
        return o.think()
    if hasattr(o, 'shape'):
        return o
    raise TypeError(o)


def ugly_diffget(params, project):
    """
        Difference between your top-down and bottom-up knowledge.
        Here's where things get interesting.
    """

    O, A, V = params
    t = project(O, A)
    v = project(V, A)

    # WE WANT THE GRADIENT OF THIS WITH RESPECT TO EVERYTHING THAT PRODUCED IT!
    return fast.dist(t,v)


def make_parameters(objects):
    for obj in objects:
        assert isinstance(obj, Object)
    return [make_parameter(o) for o in objects]

def make_parameter(o):
    params = o.params()
    if len(params) == 1:
        return slow.to_array(params)
    elif len(params) > 1:
        return slow.to_array(params)
    else:
        raise ValueError(f"No params", params)


def make_train_loop(objects, ugly_func, **ugly_kwds):

    params = make_parameters(objects)

    from jax.experimental import optimizers

    ugly_func = partial(ugly_func, **ugly_kwds)
    loss_fn = make_loss(ugly_func)
    grad_fn = value_and_grad(loss_fn)
    params = tree_think(params)

    def new_adam(step_size):
        opt_init, opt_update, get_params = optimizers.adam(step_size)
        return opt_init, opt_update, get_params

    def knowledge_generator(params):
        step_size = 1e-2
        opt_init, opt_update, get_params = new_adam(step_size)
        opt_state = opt_init(params)
        """ Note: this does the gradient updates for us, and yields the new parameters.
            It is up to us (on the outside) when we want to update them.
        """
        while True:
            loss, grads = grad_fn(params)
            grads = tree_think(grads)
            opt_state = opt_update(0, grads, opt_state)
            params = get_params(opt_state)
            request = yield params
            print(f"loss is {loss}")
            if request is not None:
                opt_init, opt_update, get_params = new_adam(request)
                opt_state = opt_init(params)
                print(f"reset step_size to {request}")

    g = knowledge_generator(params)
    g.send(None)
    return g


### tests!

Ticker  = String('Ticker')
SPY     = Ticker('SPY')
QQQ     = Ticker('QQQ')
TLT     = Ticker('TLT')
SOXX    = Ticker('SOXX')
SPXL    = Ticker('SPXL')
TQQQ    = Ticker('TQQQ')
TMF     = Ticker('TMF')
SOXL    = Ticker('SOXL')
GLD     = Ticker('GLD')
SLV     = Ticker('SLV')

Sector  = String('Sector')
MANY    = Sector('MANY')
TECH    = Sector('TECH')
SEMI    = Sector('SEMI')
GOVT    = Sector('GOVT')
NONE    = Sector('NONE')

Asset   = String('Asset')
STOCK   = Asset('STOCK')
BOND    = Asset('BOND')
COMM    = Asset('COMM')

ETF     = Boolean('ETF')
Levered = Boolean('Levered')

SPY.set(Sector, MANY)
SPY.set(Asset, STOCK)
SPY.set(ETF, True)
SPY.set(Levered, False)

QQQ.set(Sector, TECH)
QQQ.set(Asset, STOCK)
QQQ.set(ETF, True)
QQQ.set(Levered, False)

SOXX.set(Sector, SEMI)
SOXX.set(Asset, STOCK)
SOXX.set(ETF, True)
SOXX.set(Levered, False)

TLT.set(Sector, GOVT)
TLT.set(Asset, BOND)
TLT.set(ETF, True)
TLT.set(Levered, False)

SPXL.set(Sector, MANY)
SPXL.set(Asset, STOCK)
SPXL.set(ETF, True)
SPXL.set(Levered, True)

TQQQ.set(Sector, TECH)
TQQQ.set(Asset, STOCK)
TQQQ.set(ETF, True)
TQQQ.set(Levered, True)

SOXL.set(Sector, SEMI)
SOXL.set(Asset, STOCK)
SOXL.set(ETF, True)
SOXL.set(Levered, True)

TMF.set(Sector, GOVT)
TMF.set(Asset, BOND)
TMF.set(ETF, True)
TMF.set(Levered, True)

GLD.set(Sector, NONE)
GLD.set(Asset, COMM)
GLD.set(ETF, True)
GLD.set(Levered, False)

SLV.set(Sector, NONE)
SLV.set(Asset, COMM)
SLV.set(ETF, True)
SLV.set(Levered, False)

SPY.get(Sector, 'hard')
Ticker.invert(SPY)
Sector.invert(SPY)

#for object in Object.instances:
#    object.learn()

