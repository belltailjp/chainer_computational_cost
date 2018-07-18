from chainer_computational_cost.cost_calculators import register

from chainer.functions.array.concat import Concat
from chainer.functions.array.reshape import Reshape
from chainer.functions.array.resize_images import ResizeImages
from chainer.functions.array.transpose import Transpose


@register(Concat)
def calc_concat(func, in_data, **kwargs):
    """[Concat](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.concat.html)

    Concatenation is just a memory copy, hense no floating point
    operation will be conducted.
    Depending on concatenation axis, index calculation might be needed
    but chainer-computational-cost ignores such operations.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $ 0 $ |
    | mread  | $ \sum_{i} \|x_{i}\|$ |
    | mwrite | $ \sum_{i} \|x_{i}\|$ |
    | params | `axis`: concatenation axis |
    """
    size = sum([x.size for x in in_data])
    return (0, size, size, {'axis': func.axis})


@register(Reshape)
def calc_reshape(func, in_data, **kwargs):
    """[Reshape](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.reshape.html)

    Reshape operation basically neither changes nor reads the data.
    It just makes an array with same data reference with different metadata.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $ 0 $ |
    | mread  | $ 0 $ |
    | mwrite | $ 0 $ |
    | params | `in_shape`: input shape, `out_shape`: output shape |
    """
    x, = in_data
    return (0, 0, 0, {'in_shape': x.in_shape, 'out_shape': func.shape})


@register(ResizeImages)
def calc_resize(func, in_data, **kwargs):
    """[ResizeImages](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.resize_images.html) TODO: もっと書く

    Current chainer's resize operation only supports bilinear interpolation.

    For each output element,
    it refers 4 neighboring pixels from the input image.

    So the amount of memory read is 4 times the output size.

    |Item|Value|
    |:---|:---|
    |FLOPs|18 * size of output|
    |mread|4 * size of output|
    |mwrite|size of output|
    """
    x, = in_data
    batch_size, in_c = x.shape[:2]
    out_size = batch_size * in_c * func.out_H * func.out_W
    params = {'size': (func.out_H, func.out_W)}
    return (out_size * 18, out_size * 4, out_size, params)


@register(Transpose)
def calc_transpose(func, in_data, **kwargs):
    """[Transpose](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.transpose.html)

    Transpose operation is just copying memory with no FLOPs.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $ 0 $ |
    | mread  | $ \| x \| $ |
    | mwrite | $ \| x \| $ |
    | params | `axes`: transpose axes |
    """
    x, = in_data
    return (0, x.size, x.size, {'axes': func.axes})
