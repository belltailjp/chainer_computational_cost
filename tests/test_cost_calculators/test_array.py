import chainer.functions as F
import numpy as np

from chainer_computational_cost.cost_calculators import calculators


def test_concat():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    flops, mread, mwrite = calculators[type(f)](f, [x, x])

    assert flops == 0
    assert mread == 2 * (3 * 10 * 10)
    assert mwrite == 2 * (3 * 10 * 10)


def test_concat_more():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    flops, mread, mwrite = calculators[type(f)](f, [x, x, x, x])

    assert flops == 0
    assert mread == 4 * (3 * 10 * 10)
    assert mwrite == 4 * (3 * 10 * 10)


def test_reshape():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Reshape((1, -1))
    flops, mread, mwrite = calculators[type(f)](f, [x])
    assert flops == 0
    assert mread == 0
    assert mwrite == 0


def test_resize():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((15, 15))
    flops, mread, mwrite = calculators[type(f)](f, [x])

    # linear interpolation (1-dimensional):
    # for each output pixel, bring 2 neighboring pixels,
    # calc weight (3 flops; minus, minus and div),
    # get new pixel value (4 flops) -> total 9 flops
    # and do the same in another axis -> *2 -> 18flops/output_pix
    # https://en.wikipedia.org/wiki/Linear_interpolation

    # mread is not input size,
    # because for every output pixel 4 corresponding pixels need to be read
    assert flops == 3 * 15 * 15 * 18
    assert mread == 3 * 15 * 15 * 4
    assert mwrite == 3 * 15 * 15


def test_transpose():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.transpose.Transpose((0, 3, 1, 2))   # typical HWC2CHW transpose
    flops, mread, mwrite = calculators[type(f)](f, [x])

    # typical index calc cost: (ndim-1)*2/element
    assert flops == 0
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 10 * 10
