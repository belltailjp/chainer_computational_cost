import chainer.functions as F
import numpy as np

from chainer_computational_cost.cost_calculators import calculators


def test_add():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Add()
    flops, mread, mwrite, params = calculators[type(f)](f, [x, x])

    assert flops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
    assert params == dict()


def test_add_constant():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.AddConstant(x)
    flops, mread, mwrite, params = calculators[type(f)](f, [x])

    assert flops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
    assert params == dict()


def test_sub():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Sub()
    flops, mread, mwrite, params = calculators[type(f)](f, [x, x])

    assert flops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
    assert params == dict()
