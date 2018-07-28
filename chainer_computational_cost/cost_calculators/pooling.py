from chainer_computational_cost.cost_calculators import register

from chainer.functions.pooling.average_pooling_2d import AveragePooling2D
from chainer.functions.pooling.max_pooling_2d import MaxPooling2D
from chainer.functions.pooling.upsampling_2d import Upsampling2D

from chainer.utils.conv import get_conv_outsize
from chainer.utils.conv import get_deconv_outsize


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

    kh, kw = int(func.kh), int(func.kw)
    sy, sx = int(func.sy), int(func.sx)
    ph, pw = int(func.ph), int(func.pw)
    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, kh, sy, ph, cover_all=func.cover_all)
    out_w = get_conv_outsize(in_w, kw, sx, pw, cover_all=func.cover_all)

    out_size = batch_size * in_c * out_h * out_w
    flops = out_size * kw * kh

    params = {
        'k': kw if kw == kh else (kh, kw),
        's': sx if sx == sy else (sy, sx),
        'p': pw if pw == ph else (ph, pw)
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

    kh, kw = int(func.kh), int(func.kw)
    sy, sx = int(func.sy), int(func.sx)
    ph, pw = int(func.ph), int(func.pw)
    batch_size, in_c, in_h, in_w = x.shape
    out_h = get_conv_outsize(in_h, kh, sy, ph, cover_all=func.cover_all)
    out_w = get_conv_outsize(in_w, kw, sx, pw, cover_all=func.cover_all)

    out_size = batch_size * in_c * out_h * out_w
    flops = out_size * int(kw * kh - 1)

    params = {
        'k': kw if kw == kh else (kh, kw),
        's': sx if sx == sy else (sy, sx),
        'p': pw if pw == ph else (ph, pw)
    }
    return (flops, x.size, out_size, params)


@register(Upsampling2D)
def calc_upsampling_2d(func, in_data, **kwargs):
    """[Upsampling2D](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.upsampling_2d.html)

    Upsampling2D only reads the data from memory and writs to the certain
    position in the output using indices. Other pixels are filled by 0.
    Indices array has always the same shape as the input.
    Although its data type is not float but int, since their data size is
    usually the same (`float32` and `int32`), chainer-computational-cost
    ignores this difference and considers indices to be same as input.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$ 2 \| x \| $$ |
    | mwrite | $$ \| y \| $$ |
    | params | Upsampling parameter `k`, `s`, `p`, `outsize` and `cover_all` |
    """
    x, = in_data
    n, c, h, w = x.shape
    indices = func.indexes
    kh, kw = int(func.kh), int(func.kw)
    sy, sx = int(func.sy), int(func.sx)
    ph, pw = int(func.ph), int(func.pw)

    outh, outw = func.outh, func.outw
    if outh is None:
        outh = get_deconv_outsize(h, kh, sy, ph, cover_all=func.cover_all)
    if outw is None:
        outw = get_deconv_outsize(w, kw, sx, pw, cover_all=func.cover_all)

    params = {
        'k': kw if kw == kh else (kh, kw),
        's': sx if sx == sy else (sy, sx),
        'p': pw if pw == ph else (ph, pw),
        'outsize': (outh, outw),
        'cover_all': func.cover_all
    }
    return (0, x.size + indices.size, n * c * outh * outw, params)
