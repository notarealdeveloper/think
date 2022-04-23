import pytest

from think import *

def test_everything():
    a = Int(42)
    b = Int(69)
    c = a + b
    d = Add(a, b)
    e = SymbolicAdd(a, b)
    assert a.type is int
    assert b.type is int
    assert c.type is int
    assert d.type is int
    assert e.type is int
    assert jnp.allclose((a + b).think(), a.think() + b.think())
    assert jnp.allclose((a - b).think(), a.think() - b.think())

    a = Str('Hello')
    b = Str('world')
    c = a + b
    d = Add(a, b)
    e = SymbolicAdd(a, b)
    assert a.type is str
    assert b.type is str
    assert c.type is str
    assert d.type is str
    assert e.type is str
    assert jnp.allclose((a + b).think(), a.think() + b.think())

    a = Int(42)
    b = Str('cake')
    c = SymbolicAdd(a, b)

    class Integer(Type):
        def __init__(self, name):
            super().__init__(name, int)

    class String(Type):
        def __init__(self, name):
            super().__init__(name, str)

    Age = Integer('Age')
    a = Age(42)
    b = Age(69)
    assert a.w is b.w
    assert jnp.allclose((a + b).think(), a.think() + b.think())
    assert jnp.allclose((a - b).think(), a.think() - b.think())

    Ticker = String('Ticker')
    with pytest.raises(TypeError):
        a = Ticker(42)

    SPY = Ticker('SPY')
    QQQ = Ticker('QQQ')
    TLT = Ticker('TLT')
    TLH = Ticker('TLH')
    GLD = Ticker('GLD')
    SLV = Ticker('SLV')

