import chainer.functions as F
import numpy as np

from chainer.functions.connection.convolution_2d \
    import Convolution2DFunction
from chainer.functions.connection.deconvolution_2d \
    import Deconvolution2DFunction
from chainer.functions.connection.linear import LinearFunction

from helpers import calculate_cost
from helpers import require_chainer_version
from helpers import require_import


def test_conv2d_with_bias_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 10, 10)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_out, c_in, k, k).astype(np.float32)
    b = np.random.randn(c_out).astype(np.float32)
    f = Convolution2DFunction(pad=(np.int64(1), np.int64(1)))
    flops, mr, mw, params = calculate_cost(f, [x, W, b], fma_1flop=True)

    assert f.apply([x, W, b])[0].shape == (1, c_out, h_out, w_out)
    assert flops == (c_in * c_out * k * k * h_out * w_out)
    assert mr == c_in * h_in * w_in + c_out * c_in * k * k + c_out
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 1, 'p': 1, 'd': 1,
        'groups': 1, 'nobias': False
    }
    assert type(params['k']) is int
    assert type(params['s']) is int
    assert type(params['p']) is int
    assert type(params['d']) is int
    assert type(params['groups']) is int


def test_conv2d_nobias_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 10, 10)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_out, c_in, k, k).astype(np.float32)
    f = Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculate_cost(f, [x, W], fma_1flop=True)
    assert f.apply([x, W])[0].shape == (1, c_out, h_out, w_out)
    assert flops == (c_in * c_out * k * k * h_out * w_out)
    assert mr == c_in * h_in * w_in + c_out * c_in * k * k
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 1, 'p': 1, 'd': 1,
        'groups': 1, 'nobias': True
    }


def test_conv2d_with_bias_no_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 10, 10)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_out, c_in, k, k).astype(np.float32)
    b = np.random.randn(c_out).astype(np.float32)
    f = Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculate_cost(f, [x, W, b], fma_1flop=False)
    assert f.apply([x, W, b])[0].shape == (1, c_out, h_out, w_out)
    assert flops == 2 * (c_in * c_out * k * k * h_out * w_out)
    assert mr == c_in * h_in * w_in + c_out * c_in * k * k + c_out
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 1, 'p': 1, 'd': 1,
        'groups': 1, 'nobias': False
    }


def test_conv2d_nobias_no_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 10, 10)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_out, c_in, k, k).astype(np.float32)
    f = Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculate_cost(f, [x, W], fma_1flop=False)
    assert f.apply([x, W])[0].shape == (1, c_out, h_out, w_out)
    assert flops == 2 * (c_in * c_out - 1) * k * k * h_out * w_out
    assert mr == c_in * h_in * w_in + c_out * c_in * k * k
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 1, 'p': 1, 'd': 1,
        'groups': 1, 'nobias': True
    }


@require_chainer_version('4.0.0')
def test_conv2d_grouped_with_bias_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (8, 10, 10), (12, 10, 10)
    k = 3
    g = 2

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_out, c_in // g, k, k).astype(np.float32)
    b = np.random.randn(c_out).astype(np.float32)
    f = Convolution2DFunction(pad=1, groups=g)
    flops, mr, mw, params = calculate_cost(f, [x, W, b], fma_1flop=True)
    assert f.apply([x, W, b])[0].shape == (1, c_out, h_out, w_out)
    assert flops == (c_in * c_out * k * k * h_out * w_out) // g
    assert mr == c_in * h_in * w_in + c_out * c_in * k * k // g + c_out
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 1, 'p': 1, 'd': 1,
        'groups': g, 'nobias': False
    }


@require_chainer_version('4.0.0')
def test_conv2d_grouped_nobias_no_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (8, 10, 10), (12, 10, 10)
    k = 3
    g = 2

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_out, c_in // g, k, k).astype(np.float32)
    f = Convolution2DFunction(pad=1, groups=g)
    flops, mr, mw, params = calculate_cost(f, [x, W], fma_1flop=False)
    assert f.apply([x, W])[0].shape == (1, c_out, h_out, w_out)
    assert flops == 2 * g * (c_in * c_out // g**2 - 1) * k * k * h_out * w_out
    assert mr == c_in * h_in * w_in + c_out * c_in * k * k // g
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 1, 'p': 1, 'd': 1,
        'groups': g, 'nobias': True
    }


def test_deconv2d_with_bias_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 21, 21)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_in, c_out, k, k).astype(np.float32)
    b = np.random.randn(c_out).astype(np.float32)
    f = Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculate_cost(f, [x, W, b], fma_1flop=True)

    assert f.apply([x, W, b])[0].shape == (1, c_out, h_out, w_out)
    assert flops == c_in * c_out * k * k * h_in * w_in
    assert mr == c_in * h_in * w_in + c_in * c_out * k * k + c_out
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 2, 'p': 0, 'd': 1,
        'groups': 1, 'nobias': False
    }


def test_deconv2d_with_nobias_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 21, 21)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_in, c_out, k, k).astype(np.float32)
    f = Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculate_cost(f, [x, W], fma_1flop=True)
    assert f.apply([x, W])[0].shape == (1, c_out, h_out, w_out)
    assert flops == c_in * c_out * k * k * h_in * w_in
    assert mr == c_in * h_in * w_in + c_in * c_out * k * k
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 2, 'p': 0, 'd': 1,
        'groups': 1, 'nobias': True
    }


def test_deconv2d_with_bias_no_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 21, 21)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_in, c_out, k, k).astype(np.float32)
    b = np.random.randn(c_out).astype(np.float32)
    f = Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculate_cost(f, [x, W, b], fma_1flop=False)
    assert f.apply([x, W, b])[0].shape == (1, c_out, h_out, w_out)
    assert flops == 2 * c_in * c_out * k * k * h_in * w_in
    assert mr == c_in * h_in * w_in + c_in * c_out * k * k + c_out
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 2, 'p': 0, 'd': 1,
        'groups': 1, 'nobias': False
    }


def test_deconv2d_with_nobias_no_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (3, 10, 10), (12, 21, 21)
    k = 3

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_in, c_out, k, k).astype(np.float32)
    f = Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculate_cost(f, [x, W], fma_1flop=False)
    assert f.apply([x, W])[0].shape == (1, c_out, h_out, w_out)
    assert flops == 2 * c_in * c_out * k * k * h_in * w_in
    assert mr == c_in * h_in * w_in + c_in * c_out * k * k
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 2, 'p': 0, 'd': 1,
        'groups': 1, 'nobias': True
    }


@require_chainer_version('4.0.0')
def test_deconv2d_grouped_with_bias_fma():
    (c_in, h_in, w_in), (c_out, h_out, w_out) = (8, 10, 10), (12, 21, 21)
    k = 3
    g = 2

    x = np.random.randn(1, c_in, h_in, w_in).astype(np.float32)
    W = np.random.randn(c_in, c_out // g, k, k).astype(np.float32)
    b = np.random.randn(c_out).astype(np.float32)
    f = Deconvolution2DFunction(stride=2, pad=0, groups=g)
    flops, mr, mw, params = calculate_cost(f, [x, W, b], fma_1flop=True)
    assert f.apply([x, W, b])[0].shape == (1, c_out, h_out, w_out)
    assert flops == c_in * c_out * k * k * h_in * w_in // g
    assert mr == c_in * h_in * w_in + c_in * c_out * k * k // g + c_out
    assert mw == c_out * h_out * w_out
    assert params == {
        'k': k, 's': 2, 'p': 0, 'd': 1,
        'groups': 2, 'nobias': False
    }


def test_linear_nobias_fma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    f = LinearFunction()
    flops, mr, mw, params = calculate_cost(f, [x, w], fma_1flop=True)
    assert flops == 10 * 20
    assert mr == 10 + 10 * 20        # input data, and weight matrix
    assert mw == 20
    assert params == {'nobias': True}


def test_linear_nobias_no_fma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    f = LinearFunction()
    flops, mr, mw, params = calculate_cost(f, [x, w], fma_1flop=False)
    # for each output neuron, weight multiplication is applied 10 times and
    # addition (10-1) times.
    assert flops == (10 + 10 - 1) * 20
    assert mr == 10 + 10 * 20
    assert mw == 20
    assert params == {'nobias': True}


def test_linear_withbias_fma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    b = np.random.randn(20).astype(np.float32)
    f = LinearFunction()
    flops, mr, mw, params = calculate_cost(f, [x, w, b], fma_1flop=True)
    assert flops == 10 * 20 + 20
    assert mr == 10 * 20 + 10 + 20   # input data, weight matrix, and bias
    assert mw == 20
    assert params == {'nobias': False}


@require_import('chainer.functions.connection.shift.Shift')
def test_shift():
    x = np.random.randn(1, 32, 10, 10).astype(np.float32)
    f = F.connection.shift.Shift(ksize=3, dilate=1)
    flops, mr, mw, params = calculate_cost(f, [x])
    assert flops == 0     # exclude index calculation
    assert mr == x.size
    assert mw == x.size
    assert params == {'k': 3, 'd': 1}
