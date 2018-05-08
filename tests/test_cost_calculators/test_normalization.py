import chainer.functions.normalization as N
import numpy as np

from chainer_computational_cost.cost_calculators import calculators


def test_fixed_bn():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    gamma = np.random.randn(3).astype(np.float32)
    beta = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.exponential(size=(3,)).astype(np.float32)
    f = N.batch_normalization.FixedBatchNormalization()
    flops, mread, mwrite = calculators[type(f)](f, [x, gamma, beta, mean, var])

    # in test mode BN, gamma, beta, mean and var will eventually become
    # channel-wise scale and shift.
    assert flops == 3 * 10 * 10 * 2
    assert mread == 3 * 10 * 10 + (3 + 3)   # input data, scale and shift param
    assert mwrite == 3 * 10 * 10


def test_lrn():     # TODO(belltailjp): verify formula
    x = np.random.randn(1, 8, 10, 10).astype(np.float32)
    f = N.local_response_normalization.LocalResponseNormalization()
    flops, mread, mwrite = calculators[type(f)](f, [x])

    # square x, neighboring sum
    c = x.shape[1]
    s = c * f.k - (f.k // 2) * 2    # sum of k-neighbor channels (no pad)
    flops_square = x.size
    flops_neighborsum = x.size * s + x.size * 3  # including *alpha, +k, **beta
    flops_total = flops_square + flops_neighborsum + x.size
    assert flops == flops_total
    assert mread == x.size + x.shape[1] * x.shape[2] * s
    assert mwrite == x.size
