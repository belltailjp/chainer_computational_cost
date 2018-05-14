import chainer.functions as F
import numpy as np

from chainer_computational_cost.cost_calculators import calculators


def test_conv2d_with_bias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculators[type(f)](f, [x, W, b], unify_fma=True)
    assert flops == (12 * 10 * 10) * (3 * 3 * 3) + (12 * 10 * 10)
    assert mr == 3 * 10 * 10 + 12 * 3 * 3 * 3 + 12
    assert mw == 12 * 10 * 10
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 1, 'sy': 1,
        'pw': 1, 'ph': 1, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': False
    }


def test_conv2d_nobias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculators[type(f)](f, [x, W], unify_fma=True)
    assert flops == (12 * 10 * 10) * (3 * 3 * 3)
    assert mr == 3 * 10 * 10 + 12 * 3 * 3 * 3
    assert mw == 12 * 10 * 10
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 1, 'sy': 1,
        'pw': 1, 'ph': 1, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': True
    }


def test_conv2d_with_bias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculators[type(f)](f, [x, W, b], unify_fma=False)
    assert flops == 2 * (12 * 10 * 10) * (3 * 3 * 3) + (12 * 10 * 10)
    assert mr == 3 * 10 * 10 + 12 * 3 * 3 * 3 + 12
    assert mw == 12 * 10 * 10
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 1, 'sy': 1,
        'pw': 1, 'ph': 1, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': False
    }


def test_conv2d_nobias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    flops, mr, mw, params = calculators[type(f)](f, [x, W], unify_fma=False)
    assert flops == 2 * (12 * 10 * 10) * (3 * 3 * 3)
    assert mr == 3 * 10 * 10 + 12 * 3 * 3 * 3
    assert mw == 12 * 10 * 10
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 1, 'sy': 1,
        'pw': 1, 'ph': 1, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': True
    }


def test_deconv2d_with_bias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculators[type(f)](f, [x, W, b], unify_fma=True)
    assert flops == (3 * 10 * 10) * (12 * 3 * 3) + (12 * 21 * 21)
    assert mr == 3 * 10 * 10 + 3 * 12 * 3 * 3 + 12
    assert mw == 12 * 21 * 21
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 2, 'sy': 2,
        'pw': 0, 'ph': 0, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': False
    }


def test_deconv2d_nobias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculators[type(f)](f, [x, W], unify_fma=True)
    assert flops == (3 * 10 * 10) * (12 * 3 * 3)
    assert mr == 3 * 10 * 10 + 3 * 12 * 3 * 3
    assert mw == 12 * 21 * 21
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 2, 'sy': 2,
        'pw': 0, 'ph': 0, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': True
    }


def test_deconv2d_with_bias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculators[type(f)](f, [x, W, b], unify_fma=False)
    assert flops == 2 * (3 * 10 * 10) * (12 * 3 * 3) + (12 * 21 * 21)
    assert mr == 3 * 10 * 10 + 3 * 12 * 3 * 3 + 12
    assert mw == 12 * 21 * 21
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 2, 'sy': 2,
        'pw': 0, 'ph': 0, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': False
    }


def test_deconv2d_nobias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    flops, mr, mw, params = calculators[type(f)](f, [x, W], unify_fma=False)
    assert flops == 2 * (3 * 10 * 10) * (12 * 3 * 3)
    assert mr == 3 * 10 * 10 + 3 * 12 * 3 * 3
    assert mw == 12 * 21 * 21
    assert params == {
        'kw': 3, 'kh': 3, 'sx': 2, 'sy': 2,
        'pw': 0, 'ph': 0, 'dx': 1, 'dy': 1,
        'groups': 1, 'nobias': True
    }


def test_linear_nobias_unifyfma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    f = F.connection.linear.LinearFunction()
    flops, mr, mw, params = calculators[type(f)](f, [x, w], unify_fma=True)
    assert flops == 10 * 20
    assert mr == 10 + 10 * 20        # input data, and weight matrix
    assert mw == 20
    assert params == {'nobias': True}


def test_linear_nobias_nounifyfma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    f = F.connection.linear.LinearFunction()
    flops, mr, mw, params = calculators[type(f)](f, [x, w], unify_fma=False)

    # for each output neuron, weight multiplication is applied 10 times and
    # addition (10-1) times.
    assert flops == (10 + 10 - 1) * 20
    assert mr == 10 + 10 * 20
    assert mw == 20
    assert params == {'nobias': True}


def test_linear_withbias_unifyfma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    b = np.random.randn(20).astype(np.float32)
    f = F.connection.linear.LinearFunction()
    flops, mr, mw, params = calculators[type(f)](f, [x, w, b], unify_fma=True)
    assert flops == 10 * 20 + 20
    assert mr == 10 * 20 + 10 + 20   # input data, weight matrix, and bias
    assert mw == 20
    assert params == {'nobias': False}


def test_shift():
    x = np.random.randn(1, 32, 10, 10).astype(np.float32)
    f = F.connection.shift.Shift(ksize=3, dilate=1)
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert flops == 0     # exclude index calculation
    assert mread == x.size
    assert mwrite == x.size
    assert params == {'kw': 3, 'kh': 3, 'dx': 1, 'dy': 1}
