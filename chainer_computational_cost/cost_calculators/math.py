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


def calc_add(func: Add, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_add_constant(func: AddConstant, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_div(func: Div, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_div_from_constant(func: DivFromConstant, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_mul(func: Mul, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_mul_constant(func: MulConstant, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_sub(func: Sub, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)


def calc_sub_from_constant(func: SubFromConstant, in_data, **kwargs):
    return _calc(func, in_data, **kwargs)
