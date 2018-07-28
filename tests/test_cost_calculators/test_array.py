import chainer.functions as F
import numpy as np

from helpers import calculate_cost


def test_broadcast_to():
    x = np.random.randn(1, 3, 1, 1).astype(np.float32)
    f = F.array.broadcast.BroadcastTo((1, 3, 10, 10))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 3
    assert mwrite == 3 * 10 * 10
    assert params == {'shape': (1, 3, 10, 10)}


def test_concat():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    flops, mread, mwrite, params = calculate_cost(f, [x, x])
    assert flops == 0
    assert mread == 2 * (3 * 10 * 10)
    assert mwrite == 2 * (3 * 10 * 10)
    assert params == {'axis': 1}


def test_concat_more():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat(axis=2)
    flops, mread, mwrite, params = calculate_cost(f, [x, x, x, x])
    assert flops == 0
    assert mread == 4 * (3 * 10 * 10)
    assert mwrite == 4 * (3 * 10 * 10)
    assert params == {'axis': 2}


def test_reshape():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Reshape((1, -1))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 0
    assert mwrite == 0
    assert params == {'in_shape': (1, 3, 100, 100), 'out_shape': (1, -1)}


def test_resize_expand():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((15, 15))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 3 * 15 * 15 * 9
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 15 * 15
    assert params == {'size': (15, 15)}


def test_resize_shrink():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((4, 4))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 3 * 4 * 4 * 9
    assert mread == 4 * 3 * 4 * 4   # 4(neighbors)*3(channel)*out_w*out_h
    assert mwrite == 3 * 4 * 4
    assert params == {'size': (4, 4)}


def test_transpose():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.transpose.Transpose((0, 3, 1, 2))   # typical HWC2CHW transpose
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 10 * 10
    assert params == {'axes': (0, 3, 1, 2)}


def test_get_item_one():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.get_item.GetItem((0, 0, 0, 0))
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 1
    assert mwrite == 1
    assert params == {'slices': [0, 0, 0, 0]}


def test_get_item_slice():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    # x[0, 0, :, :]
    slices = (0, 0, slice(None, None, None), slice(None, None, None))
    f = F.array.get_item.GetItem(slices)
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 10 * 10
    assert mwrite == 10 * 10
    assert params == {
        'slices': [0, 0, (None, None, None), (None, None, None)]
    }


def test_get_item_slice2():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    # x[0, 0, :]
    slices = (0, 0, slice(None, None, None))
    f = F.array.get_item.GetItem(slices)
    flops, mread, mwrite, params = calculate_cost(f, [x])
    assert flops == 0
    assert mread == 10 * 10
    assert mwrite == 10 * 10
    assert params == {
        'slices': [0, 0, (None, None, None)]
    }
