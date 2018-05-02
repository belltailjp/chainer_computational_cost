import chainer.functions as F
import numpy as np

import pytest

from chainer_computational_cost.cost_calculators import *


def test_fixed_bn():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    gamma = np.random.randn(3).astype(np.float32)
    beta = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.exponential(size=(3,)).astype(np.float32)
    f = F.normalization.batch_normalization.FixedBatchNormalization()
    ops, mread, mwrite = calc_fixed_bn(f, [x, gamma, beta, mean, var])

    # in test mode BN, gamma, beta, mean and var will eventually become
    # channel-wise scale and shift.
    assert ops == 3 * 10 * 10 * 2
    assert mread == 3 * 10 * 10 + (3 + 3)   # input data, scale and shift param
    assert mwrite == 3 * 10 * 10
