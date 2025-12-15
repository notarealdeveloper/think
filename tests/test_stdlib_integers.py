#!/usr/bin/env python

from think import Digit, Decimal

decimals = [Decimal(n) for n in range(100)]

def test_decimal_training_for_instances():
    d = Decimal(78)
    d.learn()
    assert d.get(Digit[0]) == 8
    assert d.get(Digit[1]) == 7

    d = Decimal(42)
    d.learn()
    assert d.get(Digit[0]) == 2
    assert d.get(Digit[1]) == 4

def test_decimal_training_for_class():
    Decimal.learn()
    for n, N in Decimal.instances().items():
        for k, digit in enumerate(reversed(str(n))):
            assert N.get(Digit[k]) == int(digit)

