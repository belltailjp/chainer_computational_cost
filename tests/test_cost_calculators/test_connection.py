import chainer.functions as F
import numpy as np

import pytest

from chainer_computational_cost.cost_calculators import *


def test_conv2d_with_bias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    ops, mread, mwrite = calc_conv2d(f, [x, W, b], unify_fma=True)
    assert ops == (12 * 10 * 10) * (3 * 3 * 3) + (12 * 10 * 10)
    assert mread == 3 * 10 * 10 + 12 * 3 * 3 * 3 + 12
    assert mwrite == 12 * 10 * 10


def test_conv2d_nobias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    ops, mread, mwrite = calc_conv2d(f, [x, W], unify_fma=True)
    assert ops == (12 * 10 * 10) * (3 * 3 * 3)
    assert mread == 3 * 10 * 10 + 12 * 3 * 3 * 3
    assert mwrite == 12 * 10 * 10


def test_conv2d_with_bias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    ops, mread, mwrite = calc_conv2d(f, [x, W, b], unify_fma=False)
    assert ops == 2 * (12 * 10 * 10) * (3 * 3 * 3) + (12 * 10 * 10)
    assert mread == 3 * 10 * 10 + 12 * 3 * 3 * 3 + 12
    assert mwrite == 12 * 10 * 10


def test_conv2d_nobias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(12, 3, 3, 3).astype(np.float32)
    f = F.connection.convolution_2d.Convolution2DFunction(pad=1)
    ops, mread, mwrite = calc_conv2d(f, [x, W], unify_fma=False)
    assert ops == 2 * (12 * 10 * 10) * (3 * 3 * 3)
    assert mread == 3 * 10 * 10 + 12 * 3 * 3 * 3
    assert mwrite == 12 * 10 * 10


def test_deconv2d_with_bias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    ops, mread, mwrite = calc_deconv2d(f, [x, W, b], unify_fma=True)
    assert ops == (3 * 10 * 10) * (12 * 3 * 3) + (12 * 21 * 21)
    assert mread == 3 * 10 * 10 + 3 * 12 * 3 * 3 + 12
    assert mwrite == 12 * 21 * 21


def test_deconv2d_nobias_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    ops, mread, mwrite = calc_deconv2d(f, [x, W], unify_fma=True)
    assert ops == (3 * 10 * 10) * (12 * 3 * 3)
    assert mread == 3 * 10 * 10 + 3 * 12 * 3 * 3
    assert mwrite == 12 * 21 * 21


def test_deconv2d_with_bias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    ops, mread, mwrite = calc_deconv2d(f, [x, W, b], unify_fma=False)
    assert ops == 2 * (3 * 10 * 10) * (12 * 3 * 3) + (12 * 21 * 21)
    assert mread == 3 * 10 * 10 + 3 * 12 * 3 * 3 + 12
    assert mwrite == 12 * 21 * 21


def test_deconv2d_nobias_no_unifyfma():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    W = np.random.randn(3, 12, 3, 3).astype(np.float32)
    f = F.connection.deconvolution_2d.Deconvolution2DFunction(stride=2, pad=0)
    ops, mread, mwrite = calc_deconv2d(f, [x, W], unify_fma=False)
    assert ops == 2 * (3 * 10 * 10) * (12 * 3 * 3)
    assert mread == 3 * 10 * 10 + 3 * 12 * 3 * 3
    assert mwrite == 12 * 21 * 21


def test_linear_nobias_unifyfma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    f = F.connection.linear.LinearFunction()
    ops, mread, mwrite = calc_linear(f, [x, w], unify_fma=True)
    assert ops == 10 * 20
    assert mread == 10 + 10 * 20        # input data, and weight matrix
    assert mwrite == 20


def test_linear_nobias_nounifyfma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    f = F.connection.linear.LinearFunction()
    ops, mread, mwrite = calc_linear(f, [x, w], unify_fma=False)

    # for each output neuron, weight multiplication is applied 10 times and
    # addition (10-1) times.
    assert ops == (10 + 10 - 1) * 20
    assert mread == 10 + 10 * 20
    assert mwrite == 20


def test_linear_withbias_unifyfma():
    x = np.random.randn(1, 10).astype(np.float32)
    w = np.random.randn(20, 10).astype(np.float32)
    b = np.random.randn(20).astype(np.float32)
    f = F.connection.linear.LinearFunction()
    ops, mread, mwrite = calc_linear(f, [x, w, b], unify_fma=True)
    assert ops == 10 * 20 + 20
    assert mread == 10 * 20 + 10 + 20   # input data, weight matrix, and bias
    assert mwrite == 20


def test_shift():
    x = np.random.randn(1, 32, 10, 10).astype(np.float32)
    f = F.connection.shift.Shift(ksize=3, dilate=1)
    ops, mread, mwrite = calc_shift(f, [x])
    assert ops == 0     # exclude index calculation
    assert mread == x.size
    assert mwrite == x.size
