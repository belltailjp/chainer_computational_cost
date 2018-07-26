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
    ret = calculators[type(f)](f, [x, gamma, beta, mean, var])
    flops, mread, mwrite, params = ret

    # in test mode BN, gamma, beta, mean and var will eventually become
    # channel-wise scale and shift.
    assert flops == 3 * 10 * 10 * 2
    assert mread == 3 * 10 * 10 + (3 + 3)   # input data, scale and shift param
    assert mwrite == 3 * 10 * 10
    assert params == {'eps': f.eps}


def test_fixed_bn_fma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    gamma = np.random.randn(3).astype(np.float32)
    beta = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.exponential(size=(3,)).astype(np.float32)
    f = N.batch_normalization.FixedBatchNormalization()
    ret = calculators[type(f)](f, [x, gamma, beta, mean, var], fma_1flop=True)
    flops, mread, mwrite, params = ret

    # in test mode BN, gamma, beta, mean and var will eventually become
    # channel-wise scale and shift.
    assert flops == 3 * 10 * 10
    assert mread == 3 * 10 * 10 + (3 + 3)   # input data, scale and shift param
    assert mwrite == 3 * 10 * 10
    assert params == {'eps': f.eps}


def test_lrn():
    c, h, w = 8, 10, 10
    x = np.random.randn(1, c, h, w).astype(np.float32)
    f = N.local_response_normalization.LocalResponseNormalization()
    flops, mread, mwrite, params = calculators[type(f)](f, [x])

    assert flops == (6 * c - 1) * h * w
    assert mread == x.size
    assert mwrite == x.size
    assert params == {
        'n': 5, 'k': 2,
        'alpha': 0.0001, 'beta': 0.75
    }


def test_lrn_fma():
    c, h, w = 8, 10, 10
    x = np.random.randn(1, c, h, w).astype(np.float32)
    f = N.local_response_normalization.LocalResponseNormalization()
    flops, mread, mwrite, params = calculators[type(f)](f, [x], fma_1flop=True)

    assert flops == (5 * c - 1) * h * w
    assert mread == x.size
    assert mwrite == x.size
    assert params == {
        'n': 5, 'k': 2,
        'alpha': 0.0001, 'beta': 0.75
    }
