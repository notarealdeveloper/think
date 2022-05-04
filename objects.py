#!/usr/bin/env python3

__all__ = [
    'Memory',
]

import os
import glob
import pickle
from collections import Counter

import fast
from think import Object, hybridmethod


class Memory(Object):

    def __init_subclass__(cls):
        cls.num_contexts = 0
        cls.memory = {}
        cls.counts = Counter()
        cls.long_term_memory = cls.__qualname__

    def __new__(cls, obj):
        if obj in cls.memory:
            return cls.memory[obj]
        self = super().__new__(cls, obj)
        self.counts = Counter()
        self.num_contexts = 0
        cls.memory[obj] = self
        return self

    @hybridmethod
    def connect(self, others):
        self.num_contexts += 1
        for other in others:
            self.counts[other] += 1

    @hybridmethod
    def count(self, object):
        return self.counts[object]

    @hybridmethod
    def total(self):
        return sum(self.counts.values())

    @hybridmethod
    def prob(self, obj):
        return self.count(obj)/self.num_contexts

    @hybridmethod
    def probs(self):
        return Counter({k:self.prob(v) for k,v in self.counts.items()})

    @classmethod
    def probs_given(cls, b):
        return cls(b).probs()

    @classmethod
    def prob_a_given_b(cls, a, b):
        return cls(b).prob(a)

    def __getitem__(self, item):
        return self.counts[item]

    @classmethod
    def __class_getitem__(cls, item):
        return cls.counts[item]

    def connections(self):
        return tuple(sorted(self.probs().keys()))

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, dict):
        self.__dict__ = dict

    def __reduce__(self):
        tuple = (self.__class__, (self.object,), self.__getstate__())
        return tuple

    @classmethod
    def save_all(cls, directory=None):
        directory = directory or cls.long_term_memory
        for name, obj in cls.memory.items():
            pathname = f"{directory}/{name}.pkl"
            obj.save(pathname)

    @classmethod
    def load_all(cls, directory=None):
        directory = directory or cls.long_term_memory
        pathnames = sorted(glob.glob(f"{directory}/*.pkl"))
        for pathname in pathnames:
            name = os.path.basename(pathname).rstrip('.pkl')
            cls(name).load(pathname)

    def save(self, pathname=None):
        pathname = pathname or f"{cls.long_term_memory}/{self.object}.pkl"
        os.makedirs(os.path.dirname(pathname), exist_ok=True)
        with open(pathname, 'wb') as fp:
            pickle.dump(self, fp)
            print(f"Saved {self} to {pathname}")

    def load(self, pathname=None):
        pathname = pathname or f"{cls.long_term_memory}/{self.object}.pkl"
        if not os.path.exists(pathname):
            raise FileNotFoundError(pathname)
        with open(pathname, 'rb') as fp:
            self.__dict__ = pickle.load(fp).__dict__
            print(f"Loaded {self} from {pathname}")
        return self

    def most_similar(self, k=10):
        others = self.__class__.memory.values()
        if isinstance(self, str):
            self = self.__class__(self)
            t_self = self.think()
        elif isinstance(self, __class__):
            t_self = self.think()
        else:
            t_self = self
            self = None
        sims = {}
        for other in others:
            if other == self:
                continue
            t_other = other.think()
            sims[other.object] = fast.cos(t_self, t_other).item()
        pairs = sorted(sims.items(), key=lambda pair: pair[1], reverse=True)
        return pairs[:k]

    @classmethod
    def export(cls):
        G = globals()
        for k,v in cls.memory.items():
            if k in G and not isinstance(G[k], cls):
                continue
            line = f"{k}={cls.__name__}({k!r})"
            try:
                exec(line, globals())
                print(line)
            except:
                pass

