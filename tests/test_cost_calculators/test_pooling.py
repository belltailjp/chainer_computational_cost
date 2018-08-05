import chainer.functions.pooling as P
import numpy as np

from helpers import calculate_cost


def test_max_pooling():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = P.max_pooling_2d.MaxPooling2D(np.int64(2), np.int64(2),
                                      np.int64(0), cover_all=True)
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # flops is (output size) * (inside window operation)
    # when window size is 2x2, max operation is applied 2x2-1 times.
    assert flops == (3 * 50 * 50) * (2 * 2 - 1)
    assert mread == x.size
    assert mwrite == (3 * 50 * 50)
    assert params == {'k': 2, 's': 2, 'p': 0}
    assert type(params['k']) is int
    assert type(params['s']) is int
    assert type(params['p']) is int


def test_average_pooling():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = P.average_pooling_2d.AveragePooling2D(np.int64(2), np.int64(2),
                                              np.int64(0), cover_all=True)
    flops, mread, mwrite, params = calculate_cost(f, [x])

    # flops is (output size) * (inside window operation)
    # when window size is 2x2, max operation is applied 2x2-1 times.
    assert flops == (3 * 50 * 50) * ((2 * 2 - 1) + 1)
    assert mread == x.size
    assert mwrite == (3 * 50 * 50)
    assert params == {'k': 2, 's': 2, 'p': 0}
    assert type(params['k']) is int
    assert type(params['s']) is int
    assert type(params['p']) is int


def test_upsampling_2d():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    indices = np.random.randint(0, 9, (1, 3, 10, 10)).astype(np.int32)
    f = P.upsampling_2d.Upsampling2D(indices, ksize=np.int64(3),
                                     stride=np.int64(3), outsize=(30, 30))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 2 * 3 * 10 * 10
    assert mwrite == 3 * 30 * 30
    assert params == {
        'k': 3, 's': 3, 'p': 0, 'outsize': (30, 30), 'cover_all': True
    }
    assert type(params['k']) is int
    assert type(params['s']) is int
    assert type(params['p']) is int


def test_upsampling_2d_no_outsize():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    indices = np.random.randint(0, 9, (1, 3, 10, 10)).astype(np.int32)
    f = P.upsampling_2d.Upsampling2D(indices, ksize=np.int64(3),
                                     stride=np.int64(3))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 2 * 3 * 10 * 10
    assert mwrite == 3 * 28 * 28
    assert params == {
        'k': 3, 's': 3, 'p': 0, 'outsize': (28, 28), 'cover_all': True
    }
    assert type(params['k']) is int
    assert type(params['s']) is int
    assert type(params['p']) is int
