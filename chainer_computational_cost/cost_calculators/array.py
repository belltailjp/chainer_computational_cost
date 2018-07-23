from chainer_computational_cost.cost_calculators import register

from chainer.functions.array.concat import Concat
from chainer.functions.array.reshape import Reshape
from chainer.functions.array.resize_images import ResizeImages
from chainer.functions.array.transpose import Transpose


@register(Concat)
def calc_concat(func, in_data, **kwargs):
    size = sum([x.size for x in in_data])
    return (0, size, size, {'axis': func.axis})


@register(Reshape)
def calc_reshape(func, in_data, **kwargs):
    return (0, 0, 0, {'shape': func.shape})


@register(ResizeImages)
def calc_resize(func, in_data, **kwargs):
    x, = in_data
    batch_size, in_c = x.shape[:2]
    out_size = batch_size * in_c * func.out_H * func.out_W
    params = {'size': (func.out_H, func.out_W)}
    return (out_size * 18, out_size * 4, out_size, params)


@register(Transpose)
def calc_transpose(func, in_data, **kwargs):
    x, = in_data
    return (0, x.size, x.size, {'axes': func.axes})
