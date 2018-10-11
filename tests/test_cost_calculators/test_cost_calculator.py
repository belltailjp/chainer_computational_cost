import chainer

import pytest

from chainer_computational_cost.cost_calculators import register
from chainer_computational_cost.cost_calculators.cost_calculators import \
    calculators


class DummyFunction(chainer.Function):
    pass


chainer.DummyFunction = DummyFunction


def test_invalid_signature():
    with pytest.warns(UserWarning):
        @register(chainer.DummyFunction)
        def calculator_ng(func, in_data):
            pass

    # OK pattern
    @register(chainer.DummyFunction)
    def calculator_ok(func, in_data, **kwargs):
        pass
    assert chainer.DummyFunction in calculators


def test_invalid_class():
    @register('Invalid.Class')
    def calculator1(func, in_data, **kwargs):
        pass
    assert 'Invalid.Class' not in calculators

    @register('chainer.InvalidClass')
    def calculator2(func, in_data, **kwargs):
        pass
    assert 'chainer.InvalidClass' not in calculators

    # OK pattern
    @register('chainer.DummyFunction')
    def calculator3(func, in_data, **kwargs):
        pass
    assert chainer.DummyFunction in calculators
