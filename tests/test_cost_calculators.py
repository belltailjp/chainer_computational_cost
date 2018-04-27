import pytest

import chainer
import chainer.functions as F
import chainer.functions.connection
import chainer.links as L
import numpy as np

import chainer_computational_cost
from chainer_computational_cost.cost_calculators import *


def test_reshape():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Reshape((1, -1))
    ops, mread, mwrite = calc_reshape(f, [x])
    assert ops == 0
    assert mread == 0
    assert mwrite == 0


def test_activation_relu():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.ReLU()
    ops, mread, mwrite = calc_activation(f, [x])
    assert ops == x.size
    assert mread == x.size
    assert mwrite == x.size


def test_activation_sigmoid():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Sigmoid()
    ops, mread, mwrite = calc_activation(f, [x])
    assert ops == x.size
    assert mread == x.size
    assert mwrite == x.size


def test_max_pooling():     # TODO: implement more test cases
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.MaxPooling2D(2, 2, 0, cover_all=True)
    ops, mread, mwrite = calc_max_pooling2d(f, [x])

    # ops is (output size) * (inside window operation)
    # when window size is 2x2, max operation is applied 2x2-1 times.
    assert ops == (3 * 50 * 50) * (2 * 2 - 1)
    assert mread == x.size
    assert mwrite == (3 * 50 * 50)


def test_average_pooling():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.AveragePooling2D(2, 2, 0, cover_all=True)
    ops, mread, mwrite = calc_average_pooling2d(f, [x])

    # ops is (output size) * (inside window operation)
    # when window size is 2x2, max operation is applied 2x2-1 times.
    assert ops == (3 * 50 * 50) * ((2 * 2 - 1) + 1)
    assert mread == x.size
    assert mwrite == (3 * 50 * 50)


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


def test_fixed_bn():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    gamma = np.random.randn(3).astype(np.float32)
    beta = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.exponential(size=(3,)).astype(np.float32)
    f = F.normalization.batch_normalization.FixedBatchNormalization()
    ops, mread, mwrite = calc_fixed_bn(f, [x, gamma, beta, mean, var])

    # in test mode BN, gamma, beta, mean and var will eventually become
    # channel-wise scale and shift.
    assert ops == 3 * 10 * 10 * 2
    assert mread == 3 * 10 * 10 + (3 + 3)   # input data, scale and shift param
    assert mwrite == 3 * 10 * 10


def test_resize():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.resize_images.ResizeImages((15, 15))
    ops, mread, mwrite = calc_resize(f, [x])

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


def test_shift():
    x = np.random.randn(1, 32, 10, 10).astype(np.float32)
    f = F.connection.shift.Shift(ksize=3, dilate=1)
    ops, mread, mwrite = calc_shift(f, [x])
    assert ops == 0     # exclude index calculation
    assert mread == x.size
    assert mwrite == x.size


def test_transpose():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.transpose.Transpose((0, 3, 1, 2))   # typical HWC2CHW transpose
    ops, mread, mwrite = calc_transpose(f, [x])

    # typical index calc cost: (ndim-1)*2/element
    assert ops == 0
    assert mread == 3 * 10 * 10
    assert mwrite == 3 * 10 * 10


def test_concat():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    ops, mread, mwrite = calc_concat(f, [x, x])

    assert ops == 0
    assert mread == 2 * (3 * 10 * 10)
    assert mwrite == 2 * (3 * 10 * 10)


def test_concat():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.array.concat.Concat()
    ops, mread, mwrite = calc_concat(f, [x, x, x, x])

    assert ops == 0
    assert mread == 4 * (3 * 10 * 10)
    assert mwrite == 4 * (3 * 10 * 10)


def test_add():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Add()
    ops, mread, mwrite = calc_eltw_op(f, [x, x])

    assert ops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10


def test_add_constant():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.AddConstant(x)
    ops, mread, mwrite = calc_eltw_op(f, [x])

    assert ops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10


def test_sub():
    x = np.random.randn(1, 3, 10, 10).astype(np.float32)
    f = F.math.basic_math.Sub()
    ops, mread, mwrite = calc_eltw_op(f, [x, x])

    assert ops == 3 * 10 * 10
    assert mread == (3 * 10 * 10) * 2
    assert mwrite == 3 * 10 * 10
