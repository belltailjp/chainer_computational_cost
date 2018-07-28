import warnings

from chainer_computational_cost.cost_calculators import register

from chainer.functions.normalization.batch_normalization \
    import FixedBatchNormalization
from chainer.functions.normalization.l2_normalization \
    import NormalizeL2
from chainer.functions.normalization.local_response_normalization \
    import LocalResponseNormalization


@register(FixedBatchNormalization)
def calc_fixed_bn(func, in_data, **kwargs):
    """[FixedBatchNormalization](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.fixed_batch_normalization.html)

    Test-mode batch normalization.
    It consists of normalization part (using $\mu$ and $\sigma$) and
    bias part ($\\gamma$ and $\\beta$), both are composed of
    elementwise scale and shift. However this can actually be fused into single
    scale and shift operation.
    Therefore, regardless of existence of bias ($\\gamma$ and $\\beta$),
    computational cost is always $2 \|x\|$ FLOPs.

    Since scale-and-shift operation can be done by FMA,
    it becomes $\|x\|$ FLOPs if `fma_1flop` is set to `True`.

    Due to the same reason as explained above, reading learned scale and shift
    parameter is required only once (not twice) regardless of bias existence.
    Both are 1-dimensional array with $c_{\mathrm{in}}$ elements.

    | Item          | Value |
    |:--------------|:------|
    | FLOPs(FMA)    | $$ \| x \| $$ |
    | FLOPs(no-FMA) | $$ 2 \| x \| $$ |
    | mread         | $$ \|x\| + 2 c_{\mathrm{in}} $$ |
    | mwrite        | $$ \| x \| $$ |
    | params        | `eps`: epsilon for BN |
    """
    x, _, _, mean, var = in_data
    x = in_data[0]
    n_elements = len(x.flatten())
    if kwargs.get('fma_1flop'):
        flops = n_elements
    else:
        flops = n_elements * 2    # *2 <- scale and shift
    mread = n_elements + len(mean) + len(var)
    mwrite = n_elements
    return (flops, mread, mwrite, {'eps': func.eps})


@register(NormalizeL2)
def calc_normalize(func, in_data, **kwargs):
    """[NormalizeL2](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.normalize.html)

    Let us assume that `axis` is channel axis, then each spatial position has
    a $c$-dimensional vector.
    Cacululation L2-norm of this vector is, with no FMA mode,
    first it applies elementwise square in $c$ FLOPs and summation
    in $c-1$ FLOPs finally sqrt in $1$ FLOP, so in total $2c$ FLOPs.
    With FMA mode, $c$ FLOPs for square and sum, then $1$ FLOP for summing up,
    in total $c+1$ FLOPs.

    Then $\eps$ is added to the L2-norm and elementwise division is applied
    in total $c+1$ FLOPs.

    Hense, total cost for L2-normalizing an array is $(2c+c+1)wh$ FLOPs with
    no FMA mode, or $(c+1+c+1)wh$ FLOPs.

    Chainer's NormalizeL2 implementation supports `axis` to be up to 2
    elements, but it's undocumented, so chainer-computational-cost only assumes
    that axis is only one dimension.

    In the below table, 3-dimensional array with shape $(c,h,w)$ is assumed
    and the axis is channel dimension, but any other shape/axis is the same.

    | Item          | Value |
    |:--------------|:------|
    | FLOPs(FMA)    | $$ \| (2c+2)wh \| $$ when shape of $x$ is $(c,h,w)$ and `axis` is 0 |
    | FLOPs(no-FMA) | $$ \| (3c+1)wh \| $$ when shape of $x$ is $(c,h,w)$ and `axis` is 0 |
    | mread         | $$ \| x \| $$ |
    | mwrite        | $$ \| x \| $$ |
    | params        | `axis` |
    """      # NOQA
    x, = in_data
    axis = func.axis
    if type(axis) is tuple:
        if 1 < len(axis):  # it can be a tuple
            warnings.warn("chainer-computational-cost only supports axis "
                          "to be 1 element.")
        axis = axis[0]
    d_axis = x.shape[axis]
    d_rest = x.size // d_axis

    if kwargs.get('fma_1flop'):
        flops = (2 * d_axis + 2) * d_rest
    else:
        flops = (3 * d_axis + 1) * d_rest

    mread = x.size
    mwrite = x.size
    params = {'axis': axis}
    return (flops, mread, mwrite, params)


@register(LocalResponseNormalization)
def calc_lrn(func, in_data, **kwargs):
    """[LocalResponseNormalization](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.local_response_normalization.html)

    Let $c$, $h$ and $w$ be the shape of $x$, so $\|x\| = chw$
    First all the values are squared ($chw$ FLOPs).
    Then it is integrated among the channel axis
    ($(c - 1)hw$ FLOPs).
    Sum of a local response for a value can be calculated by a subtraction
    (in total $chw$ FLOPs).
    Then elementwise multiplication of $\\alpha$, addition of $k$ and
    power by $\\beta$ are conducted ($3chw$ FLOPs, or $2chw$ if
    `fma_1flop` is set to `True`, as $\\alpha$ and $k$ can be done by FMA).

    In total, FLOPs will be sum of them,
    $chw + (c-1)hw + chw + 3chw == (6c-1)hw$ FLOPs with no-FMA and
    $(5c-1)hw$ with FMA.

    | Item          | Value |
    |:--------------|:------|
    | FLOPs(FMA)    | $$ (5c-1)hw $$ |
    | FLOPs(no-FMA) | $$ (6c-1)hw $$ |
    | mread         | $$ chw == \| x \| $$ |
    | mwrite        | $$ chw == \| x \| $$ |
    | params        | `n`, `k`, `alpha` and `beta` |
    """     # NOQA
    x, = in_data

    c, h, w = x.shape[1:]
    if kwargs.get('fma_1flop'):
        flops = (5 * c - 1) * h * w
    else:
        flops = (6 * c - 1) * h * w

    mread = x.size
    mwrite = x.size
    params = {
        'n': int(func.n), 'k': int(func.k),
        'alpha': float(func.alpha), 'beta': float(func.beta)
    }
    return (flops, mread, mwrite, params)
