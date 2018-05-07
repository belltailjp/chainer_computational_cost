import chainer.functions as F
import numpy as np

import pytest

from chainer_computational_cost.cost_calculators import calculators


def test_reshape():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Reshape((1, -1))
    ops, mread, mwrite = calculators[type(f)](f, [x])
    assert ops == 0
    assert mread == 0
    assert mwrite == 0


def test_resize():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((15, 15))
    ops, mread, mwrite = calculators[type(f)](f, [x])

    # linear interpolation (1-dimensional):
    # for each output pixel, bring 2 neighboring pixels,
    # calc weight (3 ops; minus, minus and div),
    # get new pixel value (4 ops) -> total 9 ops
    # and do the same in another axis -> *2 -> 18ops/output_pix
    # https://en.wikipedia.org/wiki/Linear_interpolation

    # mread is not input size,
    # because for every output pixel 4 corresponding pixels need to be read
    assert ops == 3 * 15 * 15 * 18
    assert mread == 3 * 15 * 15 * 4
    assert mwrite == 3 * 15 * 15


def test_transpose():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.transpose.Transpose((0, 3, 1, 2))   # typical HWC2CHW transpose
    ops, mread, mwrite = calculators[type(f)](f, [x])

    # typical index calc cost: (ndim-1)*2/element
    assert ops == 0
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 10 * 10


def test_concat():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    ops, mread, mwrite = calculators[type(f)](f, [x, x])

    assert ops == 0
    assert mread == 2 * (3 * 10 * 10)
    assert mwrite == 2 * (3 * 10 * 10)


def test_concat_more():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    ops, mread, mwrite = calculators[type(f)](f, [x, x, x, x])

    assert ops == 0
    assert mread == 4 * (3 * 10 * 10)
    assert mwrite == 4 * (3 * 10 * 10)
