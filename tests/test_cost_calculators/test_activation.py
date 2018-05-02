import chainer.functions as F
import numpy as np

import pytest

from chainer_computational_cost.cost_calculators import *


def test_activation_relu():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.ReLU()
    ops, mread, mwrite = calc_activation(f, [x])
    assert ops == x.size
    assert mread == x.size
    assert mwrite == x.size


def test_activation_sigmoid():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Sigmoid()
    ops, mread, mwrite = calc_activation(f, [x])
    assert ops == x.size
    assert mread == x.size
    assert mwrite == x.size


def test_activation_prelu():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    W = np.random.randn(3).astype(np.float32)
    f = F.activation.prelu.PReLUFunction()
    ops, mread, mwrite = calc_prelu(f, [x, W])
    assert ops == x.size
    assert mread == x.size + W.size
    assert mwrite == x.size
