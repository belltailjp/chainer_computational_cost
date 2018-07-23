from chainer_computational_cost.cost_calculators import register

import chainer.cuda
from chainer.functions.math.basic_math import Add
from chainer.functions.math.basic_math import AddConstant
from chainer.functions.math.basic_math import Div
from chainer.functions.math.basic_math import DivFromConstant
from chainer.functions.math.basic_math import Mul
from chainer.functions.math.basic_math import MulConstant
from chainer.functions.math.basic_math import Sub
from chainer.functions.math.basic_math import SubFromConstant


def _calc(func, in_data, **kwargs):
    x = in_data[0]
    if hasattr(func, 'value'):  # if constant
        # to support both scalar and eltw op
        s = chainer.cuda.get_array_module(x).size(x)
        return (x.size, x.size + s, x.size, {})
    else:
        return (x.size, x.size * 2, x.size, {})


@register(Add)
def calc_add(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(AddConstant)
def calc_add_constant(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(Div)
def calc_div(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(DivFromConstant)
def calc_div_from_constant(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(Mul)
def calc_mul(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(MulConstant)
def calc_mul_constant(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(Sub)
def calc_sub(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


@register(SubFromConstant)
def calc_sub_from_constant(func, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)
