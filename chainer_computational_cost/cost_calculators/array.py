from functools import reduce

from chainer_computational_cost.cost_calculators import register

from chainer.functions.array.broadcast import BroadcastTo
from chainer.functions.array.concat import Concat
from chainer.functions.array.get_item import GetItem
from chainer.functions.array.reshape import Reshape
from chainer.functions.array.resize_images import ResizeImages
from chainer.functions.array.transpose import Transpose


@register(BroadcastTo)
def calc_broadcast(func, in_data, **kwargs):
    """[BroadcastTo](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.broadcast_to.html)

    As index calculation is ignored in chainer-computational-cost,
    broadcasting is theoretically 0 FLOPs.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$ \| x \| $$ |
    | mwrite | $$ \| y \| $$ |
    | params | `shape`: output shape |
    """
    x, = in_data
    out_size = reduce(lambda x, y: x * y, func._shape)
    return (0, x.size, out_size, {'shape': func._shape})


@register(Concat)
def calc_concat(func, in_data, **kwargs):
    """[Concat](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.concat.html)

    Concatenation is just a memory copy, hense no floating point
    operation will be conducted.
    Depending on concatenation axis, index calculation might be needed
    but chainer-computational-cost ignores such operations.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$ \sum_{i} \|x_{i}\|$$ |
    | mwrite | $$ \sum_{i} \|x_{i}\|$$ |
    | params | `axis`: concatenation axis |
    """
    size = sum([x.size for x in in_data])
    return (0, size, size, {'axis': func.axis})


@register(Reshape)
def calc_reshape(func, in_data, **kwargs):
    """[Reshape](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.reshape.html)

    Reshape operation basically neither changes nor reads the data.
    It just makes an array with same data reference with different metadata.

    If your environment cannot do in-place reshape, consider overwriting
    by a custom cost calculator (refer README.md).

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$ 0 $$ |
    | mwrite | $$ 0 $$ |
    | params | `in_shape`: input shape, `out_shape`: output shape |
    """
    x, = in_data
    return (0, 0, 0, {'in_shape': x.shape, 'out_shape': func.shape})


@register(ResizeImages)
def calc_resize(func, in_data, **kwargs):
    """[ResizeImages](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.resize_images.html)

    In bilinear resize operation, each output pixel value is calculated by
    4 neighboring source pixels in the input image.

    In order to know where the source pixel is, a few index calculations
    including floating point arithmetic is needed, but these are ignored since
    chainer-computational-cost ignores such index calculations.

    To calculate an output value from 4 source pixels,
    first 3 FLOPs is spent for horizontal interpolation from 2 source pixels,
    then another 3 FLOPs for vertical, and finally 3 FLOPs for inter-axis
    interpolation, therefore in total 9 FLOPs for each output pixel.

    As for memory access, especially in case of expanding the source image,
    duplicated memory read will happen. For example, if the input image is 8x8
    and the output size is 32x32, naively reading memory runs 4096 times,
    even though the actual size of the input is only 64.
    In order to avoid such a contradiction, chainer-computational-cost
    introduces a policy to treat such case as if it loads the entire input
    data only once.

    Conversely, in case not all the image is needed (for example input is
    32x32 and the output is 8x8, where memory read is only 128),
    chainer-computational-cost simply counts 4 memory reads for each output
    pixel.

    Either smaller number is used for memory read estimation.
    In other words, memory read is formulated as
    $ \max(4 \| y\|, \|x\|) $.

    Considering extreme cases like shrinking horizontally and expanding
    vertically, this logic should be much complicated, but for simplicity
    chainer-computational-cost only uses the above formula.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 9 \| y \| $$ |
    | mread  | $$ \min(4 \| y\|, \|x\|) $$ |
    | mwrite | $$ \| y \| $$ |
    | params | `size`: output size |
    """
    x, = in_data
    batch_size, in_c = x.shape[:2]
    out_size = batch_size * in_c * func.out_H * func.out_W
    params = {'size': (func.out_H, func.out_W)}
    return (out_size * 9, min(out_size * 4, x.size), out_size, params)


@register(Transpose)
def calc_transpose(func, in_data, **kwargs):
    """[Transpose](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.transpose.html)

    Transpose operation is just copying memory with no FLOPs.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$ \| x \| $$ |
    | mwrite | $$ \| x \| $$ |
    | params | `axes`: transpose axes |
    """
    x, = in_data
    return (0, x.size, x.size, {'axes': func.axes})


@register(GetItem)
def calc_get_item(func, in_data, **kwargs):
    """[GetItem](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.get_item.html)

    Extract part of an array. This operation is zero FLOPs.
    Most of tensor libraries have view feature, which doesn't actually create
    a new array unless necessary, but this is not considered in
    chainer-computational-cost.
    Memory read runs only for the necessary elements, so both memory
    read and write are same as the size of output tensor.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ 0 $$ |
    | mread  | $$ \| y \| $$ |
    | mwrite | $$ \| y \| $$ |
    | params | `slices`: list of slices, a slice is an int or a tuple with 3 elements |
    """     # NOQA
    x, = in_data
    y = x[func.slices]
    slices = [(s.start, s.stop, s.step) if type(s) is slice else s
              for s in func.slices]
    return (0, y.size, y.size, {'slices': slices})
