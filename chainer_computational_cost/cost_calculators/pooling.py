from chainer_computational_cost.cost_calculators import register

from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D

from chainer.utils.conv import get_conv_outsize


@register(AveragePooling2D)
def calc_average_pooling2d(func, in_data, **kwargs):
    x, = in_data

    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, func.kh, func.sy, func.ph,
                             cover_all=func.cover_all)
    out_w = get_conv_outsize(in_w, func.kw, func.sx, func.pw,
                             cover_all=func.cover_all)

    out_size = (batch_size * in_c * out_h * out_w)
    flops = out_size * ((func.kw * func.kh - 1) + 1)

    params = {
        'k': func.kw if func.kw == func.kh else (func.kh, func.kw),
        's': func.sx if func.sx == func.sy else (func.sy, func.sx),
        'p': func.pw if func.pw == func.ph else (func.ph, func.pw)
    }
    return (flops, x.size, out_size, params)


@register(MaxPooling2D)
def calc_max_pooling2d(func, in_data, **kwargs):
    x, = in_data

    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, func.kh, func.sy, func.ph,
                             cover_all=func.cover_all)
    out_w = get_conv_outsize(in_w, func.kw, func.sx, func.pw,
                             cover_all=func.cover_all)

    out_size = (batch_size * in_c * out_h * out_w)
    flops = out_size * (func.kw * func.kh - 1)

    params = {
        'k': func.kw if func.kw == func.kh else (func.kh, func.kw),
        's': func.sx if func.sx == func.sy else (func.sy, func.sx),
        'p': func.pw if func.pw == func.ph else (func.ph, func.pw)
    }
    return (flops, x.size, out_size, params)
