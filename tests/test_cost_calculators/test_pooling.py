import chainer.functions as F
import numpy as np

import pytest

from chainer_computational_cost.cost_calculators import *


def test_max_pooling():     # TODO: implement more test cases
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.MaxPooling2D(2, 2, 0, cover_all=True)
    ops, mread, mwrite = calc_max_pooling2d(f, [x])

    # ops is (output size) * (inside window operation)
    # when window size is 2x2, max operation is applied 2x2-1 times.
    assert ops == (3 * 50 * 50) * (2 * 2 - 1)
    assert mread == x.size
    assert mwrite == (3 * 50 * 50)


def test_average_pooling():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.AveragePooling2D(2, 2, 0, cover_all=True)
    ops, mread, mwrite = calc_average_pooling2d(f, [x])

    # ops is (output size) * (inside window operation)
    # when window size is 2x2, max operation is applied 2x2-1 times.
    assert ops == (3 * 50 * 50) * ((2 * 2 - 1) + 1)
    assert mread == x.size
    assert mwrite == (3 * 50 * 50)
