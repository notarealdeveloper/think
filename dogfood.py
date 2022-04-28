#!/usr/bin/env python3

import os
import fast
import slow
import think

import think
from think import Type

think.thought_dim(1024)

# Now let's implement an EnumType and get the geometry working on directories.

class EnumType(Type):

    """ A lightweight metaclass-ish that's absolutely aborable! """

    def __init__(cls, name):
        cls.memory = {}

    def __call__(cls, str):
        if str in cls.memory:
            return cls.memory[str]
        self = cls.object(str)
        cls.memory[str] = self
        return self

    def params(cls):
        return [v for v in cls.memory.values()]

    def invert(cls, object):
        keys    = [k for k,v in cls.memory.items()]
        vals    = [v.thought for k,v in cls.memory.items()]
        t       = slow.to_thought(object)
        sims    = slow.pre_attention_l1(vals, object)
        idx     = int(jnp.argmax(sims))
        key     = list(keys)[idx]
        return key

    def project(cls, object):
        return slow.attention_l1(cls.params(), object)

    def __array__(cls):
        return slow.to_array(cls.params())


Pathname = EnumType('Pathname')
Dirname  = EnumType('Dirname')
Basename = EnumType('Basename')

#pathname = Pathname('/etc/hosts')
#dirname  = Dirname('/etc')
#basename = Basename('hosts')

triples = []

for path in os.popen(f"find /etc -type f"):
    dir, base = os.path.split(path.strip())
    print(dir, base)
    pathname = Pathname(path)
    dirname  = Dirname(dir)
    basename = Basename(base)
    triples.append([pathname, dirname, basename])
    pathname.set(Dirname, dirname)
    pathname.set(Basename, basename)
    print(path)

for pathname, dirname, basename in triples:
    dirname_feeling  = pathname.get(Dirname, 'hard')
    basename_feeling = pathname.get(Basename, 'hard')
    print(dirname == dirname_feeling)
    print(basename == basename_feeling)

