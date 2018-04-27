import math

import chainer
from chainer.utils.conv import get_conv_outsize, get_deconv_outsize


def calc_eltw_op(function, in_data, **kwargs):
    x = in_data[0]
    return (x.size, x.size * 2, x.size)

def calc_linear(function, in_data, **kwargs):
    x, W = in_data[:2]
    batch_size, in_c = x.shape
    out_c, _ = W.shape

    if kwargs['unify_fma']:
        ops = batch_size * in_c * out_c
    else:
        ops = batch_size * (in_c + in_c - 1) * out_c

    mread = x.size + W.size
    mwrite = out_c

    if len(in_data) == 3:
        b = in_data[2]
        ops += b.size
        mread += b.size

    return (ops, mread, mwrite)

def calc_conv2d(function, in_data, **kwargs):
    x, W = in_data[:2]
    b = in_data[2] if len(in_data) == 3 else None

    batch_size, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = W.shape
    g = function.groups

    out_h = get_conv_outsize(in_h, kh, function.sy, function.ph,
                             cover_all=function.cover_all, d=function.dy)
    out_w = get_conv_outsize(in_w, kw, function.sx, function.pw,
                             cover_all=function.cover_all, d=function.dx)

    ops = in_c * int(math.ceil(out_c / g)) * kw * kh * out_w * out_h
    if not kwargs['unify_fma']:
        ops *= 2

    mread = x.size + W.size
    mwrite = batch_size * out_c * out_h * out_w
    if b is not None:
        ops += batch_size * out_c * out_w * out_h
        mread += b.size

    return (ops * batch_size, mread, mwrite)

def calc_deconv2d(function, in_data, **kwargs):
    x, W = in_data[:2]
    b = in_data[2] if len(in_data) == 3 else None

    batch_size, in_c, in_h, in_w = x.shape
    _, out_c, kh, kw = W.shape
    g = function.groups

    out_h = get_deconv_outsize(in_h, kh, function.sy,
                               function.ph, d=function.dy)
    out_w = get_deconv_outsize(in_w, kw, function.sx,
                               function.pw, d=function.dx)

    ops = in_c * int(math.ceil(out_c / g)) * kw * kh * in_w * in_h
    if not kwargs['unify_fma']:
        ops *= 2

    mread = x.size + W.size
    mwrite = batch_size * out_c * out_h * out_w
    if b is not None:
        ops += batch_size * out_c * out_w * out_h
        mread += b.size

    return (ops * batch_size, mread, mwrite)

def calc_fixed_bn(function, in_data, **kwargs):
    x, _, _, mean, var = in_data
    x = in_data[0]
    n_elements = len(x.flatten())
    ops = n_elements * 2    # *2 <- scale and shift
    mread = n_elements + len(mean) + len(var)
    mwrite = n_elements
    return (ops, mread, mwrite)

def calc_activation(function, in_data, **kwargs):
    x, = in_data
    ops = x.size
    return (ops, ops, ops)

def calc_reshape(function, in_data, **kwargs):
    size = in_data[0].size
    return (0, 0, 0)

def calc_max_pooling2d(function, in_data, **kwargs):
    x, = in_data

    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, function.kh, function.sy, function.ph,
                             cover_all=function.cover_all)
    out_w = get_conv_outsize(in_w, function.kw, function.sx, function.pw,
                             cover_all=function.cover_all)

    out_size = (batch_size * in_c * out_h * out_w)
    ops = out_size * (function.kw * function.kh - 1)

    return (ops, x.size, out_size)

def calc_average_pooling2d(function, in_data, **kwargs):
    x, = in_data

    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, function.kh, function.sy, function.ph,
                             cover_all=function.cover_all)
    out_w = get_conv_outsize(in_w, function.kw, function.sx, function.pw,
                             cover_all=function.cover_all)

    out_size = (batch_size * in_c * out_h * out_w)
    ops = out_size * ((function.kw * function.kh - 1) + 1)

    return (ops, x.size, out_size)

def calc_resize(function, in_data, **kwargs):
    x, = in_data
    batch_size, in_c = x.shape[:2]
    out_size = batch_size * in_c * function.out_H * function.out_W
    return (out_size * 18, out_size * 4, out_size)

def calc_shift(function, in_data, **kwargs):
    x, = in_data
    return (0, x.size, x.size)

def calc_transpose(function, in_data, **kwargs):
    x, = in_data
    return (0, x.size, x.size)

def calc_concat(function, in_data, **kwargs):
    size = sum([x.size for x in in_data])
    return (0, size, size)


calculators = {
    '_ + _': calc_eltw_op,
    '_ - _': calc_eltw_op,
    '_ * _': calc_eltw_op,
    '_ / _': calc_eltw_op,
    'LinearFunction': calc_linear,
    'Convolution2DFunction': calc_conv2d,
    'ReLU': calc_activation,
    'Sigmoid': calc_activation,
    'FixedBatchNormalization': calc_fixed_bn,
    'Reshape': calc_reshape,
    'MaxPooling2D': calc_max_pooling2d,
    'AveragePooling2D': calc_average_pooling2d,
    'ResizeImages': calc_resize,
    'Concat': calc_concat,
    'Shift': calc_shift
}

