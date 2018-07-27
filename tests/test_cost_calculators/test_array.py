import chainer.functions as F
import numpy as np

from chainer_computational_cost.cost_calculators import calculators


def test_broadcast_to():
    x = np.random.randn(1, 3, 1, 1).astype(np.float32)
    f = F.array.broadcast.BroadcastTo((1, 3, 10, 10))
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    assert flops == 0
    assert mread == 3
    assert mwrite == 3 * 10 * 10
    assert params == {'shape': (1, 3, 10, 10)}


def test_concat():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    flops, mread, mwrite, params = calculators[type(f)](f, [x, x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    assert flops == 0
    assert mread == 2 * (3 * 10 * 10)
    assert mwrite == 2 * (3 * 10 * 10)
    assert params == {'axis': 1}


def test_concat_more():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat(axis=2)
    flops, mread, mwrite, params = calculators[type(f)](f, [x, x, x, x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    assert flops == 0
    assert mread == 4 * (3 * 10 * 10)
    assert mwrite == 4 * (3 * 10 * 10)
    assert params == {'axis': 2}


def test_reshape():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Reshape((1, -1))
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    assert flops == 0
    assert mread == 0
    assert mwrite == 0
    assert params == {'in_shape': (1, 3, 100, 100), 'out_shape': (1, -1)}


def test_resize_expand():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((15, 15))
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    assert flops == 3 * 15 * 15 * 9
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 15 * 15
    assert params == {'size': (15, 15)}


def test_resize_shrink():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((4, 4))
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    assert flops == 3 * 4 * 4 * 9
    assert mread == 4 * 3 * 4 * 4   # 4(neighbors)*3(channel)*out_w*out_h
    assert mwrite == 3 * 4 * 4
    assert params == {'size': (4, 4)}


def test_transpose():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.transpose.Transpose((0, 3, 1, 2))   # typical HWC2CHW transpose
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert type(flops) is int and type(mread) is int and type(mwrite) is int
    assert type(params) is dict

    # typical index calc cost: (ndim-1)*2/element
    assert flops == 0
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 10 * 10
    assert params == {'axes': (0, 3, 1, 2)}
