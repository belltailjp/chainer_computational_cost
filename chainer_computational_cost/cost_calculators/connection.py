from chainer_computational_cost.cost_calculators import register

from chainer.functions.connection.convolution_2d \
    import Convolution2DFunction
from chainer.functions.connection.deconvolution_2d \
    import Deconvolution2DFunction
from chainer.functions.connection.linear import LinearFunction
from chainer.functions.connection.shift import Shift

from chainer.utils.conv import get_conv_outsize
from chainer.utils.conv import get_deconv_outsize


@register(Convolution2DFunction)
def calc_conv2d(func, in_data, **kwargs):
    """[Convolution2DFunction](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.convolution_2d.html)

    Convolution operator essentially calculates an output value by convolving
    the input feature map by a corresponding filter whose size is
    $k_{h} k_{w} {c_{\mathrm{in}}}$.
    The computational cost is $2 k_{h} k_{w} {c_{\mathrm{in}}} - 1$ FLOPs
    for each output pixel. Including bias, it becomes simply
    $2 k_{h} k_{w} {c_{\mathrm{in}}}$.
    So in total $ 2 k_{h} k_{w} {c_{\mathrm{in}}} h_{\mathrm{out}} w_{\mathrm{out}} c_{\mathrm{out}} $.
    Here, output size $h_{\mathrm{out}}$ and $w_{\mathrm{out}}$ can be
    calculated by
    [chainer.utils.get_conv_outsize](https://docs.chainer.org/en/v4.3.0/reference/util/generated/chainer.utils.get_conv_outsize.html).

    If there is no bias, it will be
    $ (2 k_{h} k_{w} {c_{\mathrm{in}}}-1) h_{\mathrm{out}} w_{\mathrm{out}} c_{\mathrm{out}} $ FLOPs.

    In case of grouped convolution, it can be considered as just concatenating
    result of convolution whose input has $c_{\mathrm{in}}/G$ channels and
    output is $c_{\mathrm{out}}/G$ channels.
    Each group costs $ 2 k_{h} k_{w} c_{\mathrm{in}} h_{\mathrm{out}} w_{\mathrm{out}} c_{\mathrm{out}}/G^2 $ FLOPs/group,
    so in total $ 2 k_{h} k_{w} c_{\mathrm{in}} h_{\mathrm{out}} w_{\mathrm{out}} c_{\mathrm{out}} / G $ FLOPs.

    If `fma_1flop` is set to `True`,
    it will be $ k_{h} k_{w} c_{\mathrm{in}} h_{\mathrm{out}} w_{\mathrm{out}} c_{\mathrm{out}} / G $ FLOPs.
    Exsistence of bias does not matter this case.

    Dilated convolution does not change theoretical computational cost
    explained above, although it usually significantly affects to the actual
    performance.

    Although a source pixel can be read multiple times in the most native
    convolution implementation, chainer-computational-cost counts them only
    once, therefore the memory read is counted as if the entire input data and
    parameters (weights, biases) are loaded from memory only at once.
    Padding is ignored, too.

    FMA option can be switched by `fma_1flop: bool` keyword argument specified
    to ComputationalCostHook.

    | Item                | Value |
    |:--------------------|:------|
    | FLOPs(FMA)          | $$ c_{\mathrm{in}} c_{\mathrm{out}} k_{h} k_{w} h_{\mathrm{out}} w_{\mathrm{out}} / G $$ |
    | FLOPs(no-FMA)       | $$ 2 c_{\mathrm{in}} c_{\mathrm{out}} k_{h} k_{w} h_{\mathrm{out}} w_{\mathrm{out}} / G $$ |
    | FLOPs(no-FMA nobias)| $$ G(2 (c_{\mathrm{in}}/G) (c_{\mathrm{out}}/G)-1) k_{h} k_{w} h_{\mathrm{out}} w_{\mathrm{out}} $$ |
    | mread               | $$\| x \| + \| W \| + \| b \|$$ |
    | mwrite              | $$c_{\mathrm{out}} h_{\mathrm{out}} w_{\mathrm{out}}$$ |
    | params              | conv parameters `k`, `s`, `p`, `d`, `groups`, `nobias` |
    """  # NOQA
    x, W = in_data[:2]
    b = in_data[2] if len(in_data) == 3 else None

    batch_size, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = W.shape
    g = func.groups
    sy, sx = int(func.sy), int(func.sx)
    ph, pw = int(func.ph), int(func.pw)
    dy, dx = int(func.dy), int(func.dx)

    out_h = get_conv_outsize(in_h, kh, sy, ph, cover_all=func.cover_all, d=dy)
    out_w = get_conv_outsize(in_w, kw, sx, pw, cover_all=func.cover_all, d=dx)

    if kwargs.get('fma_1flop'):
        flops = in_c * (out_c // g) * kw * kh * out_w * out_h
    else:
        if b is not None:
            flops = 2 * in_c * out_c * kw * kh * out_w * out_h // g
        else:
            flops = 2 * ((in_c // g) * (out_c // g) - 1) * \
                kw * kh * out_w * out_h * g

    mread = x.size + W.size
    mwrite = batch_size * out_c * out_h * out_w
    if b is not None:
        mread += b.size

    params = {
        'k': (kw if kw == kh else (kh, kw)),
        's': (sx if sx == sy else (sy, sx)),
        'p': (pw if pw == ph else (ph, pw)),
        'd': (dx if dx == dy else (dy, dx)),
        'groups': func.groups, 'nobias': b is None
    }
    return (flops * batch_size, mread, mwrite, params)


@register(Deconvolution2DFunction)
def calc_deconv2d(func, in_data, **kwargs):
    """[Deconvolution2DFunction](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.deconvolution_2d.html)

    Deconvolution, also as known as transposed convolution, can be thought as
    going backward to the convolution operation.
    Each input pixel is multiplied to convolution filter kernel
    ($(c_{\mathrm{out}}, k_h, k_w)$).
    Its result is summed up on the output tensor, among adjacent result and
    result of other filters (there are $c_{\mathrm{in}}$ filters),
    then bias is added if exists.

    In order to understand the behavior of this operation and why it is called
    "transposed" convolution, these materials are helpful.
    * [Up-sampling with Transposed Convolution - Towards Data Science](https://towardsdatascience.com/9ae4f2df52d0)
    * [Convolution arithmetic tutorial - Theano 1.0.0 documentation](http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html)

    The theoretical computational cost is
    $2 c_{\mathrm{out}} k_h k_w h_{\mathrm{in}} w_{\mathrm{in}} c_{\mathrm{in}}$FLOPs.

    In case of `groups` is not 1, similarly to convolution, it becomes
    $2 c_{\mathrm{out}} k_h k_w h_{\mathrm{in}} w_{\mathrm{in}} c_{\mathrm{in}} / G$ FLOPs.

    | Item                | Value |
    |:--------------------|:------|
    | FLOPs(FMA)          | $$ 2 c_{\mathrm{in}} c_{\mathrm{out}} k_h k_w h_{\mathrm{in}} w_{\mathrm{in}} / G $$ |
    | FLOPs(no-FMA)       | $$ c_{\mathrm{in}} c_{\mathrm{out}} k_h k_w h_{\mathrm{in}} w_{\mathrm{in}} / G $$ |
    | mread               | $$\| x \| + \| W \| + \| b \|$$ |
    | mwrite              | $$c_{\mathrm{out}} h_{\mathrm{out}} w_{\mathrm{out}}$$ |
    | params              | conv parameters `k`, `s`, `p`, `d`, `groups`, `nobias` |
    """     # NOQA
    x, W = in_data[:2]
    b = in_data[2] if len(in_data) == 3 else None

    batch_size, in_c, in_h, in_w = x.shape
    _, out_c, kh, kw = W.shape
    g = func.groups
    # here, out_c obtained from W is already grouped
    # so the real output channels is out_c * g

    out_h = get_deconv_outsize(in_h, kh, func.sy, func.ph, d=func.dy)
    out_w = get_deconv_outsize(in_w, kw, func.sx, func.pw, d=func.dx)
    print(out_c, out_h, out_w)

    flops = in_c * out_c * kw * kh * in_w * in_h
    if not kwargs.get('fma_1flop'):
        flops *= 2

    mread = x.size + W.size
    mwrite = batch_size * (out_c * g) * out_h * out_w
    if b is not None:
        mread += b.size

    params = {
        'k': (kw if kw == kh else (kh, kw)),
        's': (func.sx if func.sx == func.sy else (func.sy, func.sx)),
        'p': (func.pw if func.pw == func.ph else (func.ph, func.pw)),
        'd': (func.dx if func.dx == func.dy else (func.dy, func.dx)),
        'groups': func.groups,
        'nobias': b is None
    }
    return (flops * batch_size, mread, mwrite, params)


@register(LinearFunction)
def calc_linear(func, in_data, **kwargs):
    """[LinearFunction](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.linear.html)

    Let $n_{\mathrm{in}}$ be the input size and $n_{\mathrm{out}}$ be the
    output size.
    Each output value is calculated by $n_{\mathrm{in}}$ product and
    $n_{\mathrm{in}} - 1$ sum operations.
    So, in case `fma_1flop==False`, $2 n_{\mathrm{in}} - 1$ FLOPs/element,
    or $2 * n_{\mathrm{in}}$ if there is bias.
    In FMA mode $n_{\mathrm{in}}$ FLOPs (regardress of existence of bias).

    FMA option can be switched by `fma_1flop: bool` keyword argument specified
    to ComputationalCostHook.

    | Item                  | Value |
    |:----------------------|:------|
    | FLOPs(FMA)            | $$\| n_{\mathrm{in}} n_{\mathrm{out}} \|$$ |
    | FLOPs(no-FMA)         | $$\| 2 n_{\mathrm{in}} n_{\mathrm{out}} \|$$ |
    | FLOPs(no-FMA no-bias) | $$\| (2 n_{\mathrm{in}} - 1) n_{\mathrm{out}} \|$$ |
    | mread                 | $$\|x\| + \|W\| + \|b\|$$, where $W$ and $b$ are learned parameter |
    | mwrite                | $$\|y\|$$ |
    | params                | `nobias` |
    """     # NOQA
    x, W = in_data[:2]
    batch_size, in_c = x.shape
    out_c, _ = W.shape

    if kwargs.get('fma_1flop'):
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


@register(Shift)
def calc_shift(func, in_data, **kwargs):
    """[Shift](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.shift.html)

    Shift only conducts memory read, index calculation and memory write.
    There might be unnecessary memory read around corners, but for simplicity
    chainer-computational-cost treats it as just reading the entire data.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$\| x \|$$ |
    | mwrite | $$\| x \|$$ |
    | params | shift parameters `k` and `d` |
    """
    x, = in_data
    params = {
        'k': (func.kw if func.kw == func.kh else (func.kh, func.kw)),
        'd': (func.dx if func.dx == func.dy else (func.dy, func.dx)),
    }
    return (0, x.size, x.size, params)
