#!/usr/bin/env python3

__all__ = [
    'Decimal',
    'Binary',
]

import jax.numpy as jnp
from think import Int, Thought


class Numeral(Int):

    def __init_subclass__(cls):
        cls.thought = Thought()
        cls.memory = {}

    def __new_context__(cls, m, t=None):
        """ Ordinal basis implementation """
        n = cls.Item
        a = cls[n].think()
        b = cls[n+1].think()
        θ = cls.theta(m)
        t = a*jnp.cos(θ) + b*jnp.sin(θ)
        self = super().__new__(cls, m, t=t)
        return self

    def theta(cls, m):
        M = len(cls.__instances__)
        return (m/M)*(jnp.pi/2)


class Digit(Numeral):
    __instances__ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class Bit(Numeral):
    __instances__ = [0, 1]


class Decimal(Int):
    def __init__(self, n):
        digits = str(n)
        for slot, digit in enumerate(reversed(digits)):
            digit = int(digit)
            self.set(Digit[slot], digit)


class Binary(Int):
    def __init__(self, n):
        digits = bin(n)[2:]
        for slot, digit in enumerate(reversed(digits)):
            digit = int(digit)
            self.set(Bit[slot], digit)


if __name__ == '__main__':

    tentens    = Digit[2](10)
    onehundred = Digit[3](0)
    assert jnp.allclose(tentens.think(), onehundred.think())

    import slow
    import seaborn as sns

    d = {}
    N = 1000
    for n, m in itertools.product(range(N), range(N)):
        nkey = str(n).zfill(len(str(N))-1)
        mkey = str(m).zfill(len(str(N))-1)
        d[nkey, mkey] = fast.cos(Decimal(n).think(), Decimal(m).think()).item()
        d[mkey, nkey] = d[nkey, mkey]
    df = pd.Series(d.values(), index=d.keys()).unstack()
    sns.set(font_scale=0.1)
    c = sns.heatmap(df, xticklabels=1, yticklabels=1)
    c.figure.set_figwidth(30)
    c.figure.set_figheight(20)
    plt.title(f"Pairwise similarities of numbers from 0 to {N} in an ordinal basis", fontsize=42)
    plt.savefig(f"pairwise-similarities-of-numbers-from-0-to-{N}-in-an-ordinal-basis.png")
    plt.close('all')

