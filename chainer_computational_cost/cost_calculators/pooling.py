from chainer_computational_cost.cost_calculators import register

from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D

from chainer.utils.conv import get_conv_outsize


@register(AveragePooling2D)
def calc_average_pooling2d(func, in_data, **kwargs):
    """[AveragePooling2D](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.average_pooling_2d.html)

    Each output pixel is calculated by averaging $k * k$ elements from the
    input ($k*k$ FLOPs). Output size is calculated by
    [chainer.utils.get_conv_outsize](https://docs.chainer.org/en/v4.3.0/reference/util/generated/chainer.utils.get_conv_outsize.html).

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ \| y \| k_{\mathrm{w}} k_{\mathrm{h}} $$ |
    | mread  | $$\| x \|$$ |
    | mwrite | $$\| y \|$$ |
    | params | AvgPooling parameter `k`, `s` and `p` |
    """
    x, = in_data

    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, func.kh, func.sy, func.ph,
                             cover_all=func.cover_all)
    out_w = get_conv_outsize(in_w, func.kw, func.sx, func.pw,
                             cover_all=func.cover_all)

    out_size = (batch_size * in_c * out_h * out_w)
    flops = out_size * func.kw * func.kh

    params = {
        'k': func.kw if func.kw == func.kh else (func.kh, func.kw),
        's': func.sx if func.sx == func.sy else (func.sy, func.sx),
        'p': func.pw if func.pw == func.ph else (func.ph, func.pw)
    }
    return (flops, x.size, out_size, params)


@register(MaxPooling2D)
def calc_max_pooling2d(func, in_data, **kwargs):
    """[MaxPooling2D](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.max_pooling_2d.html)

    Each output pixel is calculated by taking max of $k * k$ elements from the
    input ($k*k - 1$ FLOPs). Output size is calculated by
    [chainer.utils.get_conv_outsize](https://docs.chainer.org/en/v4.3.0/reference/util/generated/chainer.utils.get_conv_outsize.html).

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ \| y \| (k_{\mathrm{w}} k_{\mathrm{h}} - 1) $$ |
    | mread  | $$\| x \|$$ |
    | mwrite | $$\| y \|$$ |
    | params | AvgPooling parameter `k`, `s` and `p` |
    """
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
