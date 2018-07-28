import chainer.functions as F
import numpy as np

from helpers import calculate_cost


def test_add():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Add()
    flops, mread, mwrite, params = calculate_cost(f, [x, x])
    assert flops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
    assert params == dict()


def test_add_multiple():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Add()
    n_array = 10
    flops, mread, mwrite, params = calculate_cost(f, [x] * n_array)
    assert flops == (n_array - 1) * 3 * 10 * 10
    assert mread == n_array * (3 * 10 * 10)
    assert mwrite == 3 * 10 * 10
    assert params == dict()


def test_add_constant():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.AddConstant(x)
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
    assert params == dict()


def test_sub():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Sub()
    flops, mread, mwrite, params = calculate_cost(f, [x, x])
    assert flops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
    assert params == dict()


def test_max_noaxis():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.Max()
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 3 * 10 * 10 - 1
    assert mread == 3 * 10 * 10
    assert mwrite == 1
    assert params == {'axis': None}


def test_max_axis():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.Max(axis=2)
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 3 * (10 - 1) * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 10
    assert params == {'axis': (2,)}


def test_max_axes():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.Max(axis=(1, 2))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == (3 - 1) * 10 * 10 + (10 - 1) * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 10
    assert params == {'axis': (1, 2)}


def test_max_axes_rev():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.Max(axis=(2, 1))
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # Completely equivament to axis=(1, 2) case
    assert flops == (3 - 1) * 10 * 10 + (10 - 1) * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 10
    assert params == {'axis': (2, 1)}


def test_max_all_axes():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.Max(axis=(0, 1, 2, 3))
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # Completely equivament to axis=None case
    assert flops == 3 * 10 * 10 - 1
    assert mread == 3 * 10 * 10
    assert mwrite == 1
    assert params == {'axis': (0, 1, 2, 3)}


def test_argmax():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.ArgMax(axis=1)
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # exactly same as min/max (axis=1)
    assert flops == (3 - 1) * 10 * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 10 * 10
    assert params == {'axis': 1}


def test_argmin():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.minmax.ArgMin(axis=1)
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # exactly same as min/max
    assert flops == (3 - 1) * 10 * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 10 * 10
    assert params == {'axis': 1}


def test_sum():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.sum.Sum(axis=1)
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # exactly same as min/max
    assert flops == (3 - 1) * 10 * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 10 * 10
    assert params == {'axis': (1,)}


def test_sum_axes():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.sum.Sum(axis=(1, 2))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == (3 - 1) * 10 * 10 + (10 - 1) * 10
    assert mread == 3 * 10 * 10
    assert mwrite == 10
    assert params == {'axis': (1, 2)}


def test_sum_noaxis():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.sum.Sum()
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # exactly same as min/max
    assert flops == 3 * 10 * 10 - 1
    assert mread == 3 * 10 * 10
    assert mwrite == 1
    assert params == {'axis': None}
