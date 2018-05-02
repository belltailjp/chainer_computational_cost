from chainer.utils.conv import get_conv_outsize, get_deconv_outsize


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
