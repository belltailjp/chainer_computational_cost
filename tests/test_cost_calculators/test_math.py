import chainer.functions as F
import numpy as np

import pytest

from chainer_computational_cost.cost_calculators import *


def test_add():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Add()
    ops, mread, mwrite = calc_eltw_op(f, [x, x])

    assert ops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10


def test_add_constant():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.AddConstant(x)
    ops, mread, mwrite = calc_eltw_op(f, [x])

    assert ops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10


def test_sub():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Sub()
    ops, mread, mwrite = calc_eltw_op(f, [x, x])

    assert ops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10