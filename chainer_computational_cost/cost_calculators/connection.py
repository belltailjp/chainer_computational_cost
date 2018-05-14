import math

from chainer.functions.connection.convolution_2d \
    import Convolution2DFunction
from chainer.functions.connection.deconvolution_2d \
    import Deconvolution2DFunction
from chainer.functions.connection.linear import LinearFunction
from chainer.functions.connection.shift import Shift

from chainer.utils.conv import get_conv_outsize
from chainer.utils.conv import get_deconv_outsize


def calc_conv2d(func: Convolution2DFunction, in_data, **kwargs):
    x, W = in_data[:2]
    b = in_data[2] if len(in_data) == 3 else None

    batch_size, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = W.shape
    g = func.groups

    out_h = get_conv_outsize(in_h, kh, func.sy, func.ph,
                             cover_all=func.cover_all, d=func.dy)
    out_w = get_conv_outsize(in_w, kw, func.sx, func.pw,
                             cover_all=func.cover_all, d=func.dx)

    flops = in_c * int(math.ceil(out_c / g)) * kw * kh * out_w * out_h
    if not kwargs['unify_fma']:
        flops *= 2

    mread = x.size + W.size
    mwrite = batch_size * out_c * out_h * out_w
    if b is not None:
        flops += batch_size * out_c * out_w * out_h
        mread += b.size

    params = {
        'kw': kw, 'kh': kh,
        'sx': func.sx, 'sy': func.sy,
        'pw': func.pw, 'ph': func.ph,
        'dx': func.dx, 'dy': func.dy,
        'groups': func.groups,
        'nobias': b is None
    }
    return (flops * batch_size, mread, mwrite, params)


def calc_deconv2d(func: Deconvolution2DFunction, in_data, **kwargs):
    x, W = in_data[:2]
    b = in_data[2] if len(in_data) == 3 else None

    batch_size, in_c, in_h, in_w = x.shape
    _, out_c, kh, kw = W.shape
    g = func.groups

    out_h = get_deconv_outsize(in_h, kh, func.sy,
                               func.ph, d=func.dy)
    out_w = get_deconv_outsize(in_w, kw, func.sx,
                               func.pw, d=func.dx)

    flops = in_c * int(math.ceil(out_c / g)) * kw * kh * in_w * in_h
    if not kwargs['unify_fma']:
        flops *= 2

    mread = x.size + W.size
    mwrite = batch_size * out_c * out_h * out_w
    if b is not None:
        flops += batch_size * out_c * out_w * out_h
        mread += b.size

    params = {
        'kw': kw, 'kh': kh,
        'sx': func.sx, 'sy': func.sy,
        'pw': func.pw, 'ph': func.ph,
        'dx': func.dx, 'dy': func.dy,
        'groups': func.groups,
        'nobias': b is None
    }
    return (flops * batch_size, mread, mwrite, params)


def calc_linear(func: LinearFunction, in_data, **kwargs):
    x, W = in_data[:2]
    batch_size, in_c = x.shape
    out_c, _ = W.shape

    if kwargs['unify_fma']:
        flops = batch_size * in_c * out_c
    else:
        flops = batch_size * (in_c + in_c - 1) * out_c

    mread = x.size + W.size
    mwrite = out_c

    if len(in_data) == 3:
        b = in_data[2]
        flops += b.size
        mread += b.size

    params = {'nobias': len(in_data) < 3}
    return (flops, mread, mwrite, params)


def calc_shift(func: Shift, in_data, **kwargs):
    x, = in_data
    params = {
        'kw': func.kw, 'kh': func.kh,
        'dx': func.dx, 'dy': func.dy
    }
    return (0, x.size, x.size, params)
