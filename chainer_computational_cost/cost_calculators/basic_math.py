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


def _calc(func, xs, **kwargs):
    if hasattr(func, 'value'):  # if constant
        # to support both scalar and eltw op
        assert len(xs) == 1
        x = xs[0]
        s = chainer.cuda.get_array_module(func.value).size(x)
        return (x.size, x.size + s, x.size, {})
    else:
        return (x.size, x.size * 2, x.size, {})


@register(Add)
def calc_add(func, in_data, **kwargs):
    """[Add](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    Elementwise Add operation.
    In case shape of inputs are different, broadcast are conducted accordingly.
    The output memory size is the largest size of the input.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $ 0 $ |
    | mread  | $ \sum_{i} \| x_{i} \| $ |
    | mwrite | $ \max_{i}(\| x_{i} \|) $ |
    | params | N/A |
    """
    return _calc(func, in_data, **kwargs)


@register(AddConstant)
def calc_add_constant(func, in_data, **kwargs):
    """AddConstant

    Elementwise Add operation with constant(s).
    FLOPs is same as input element size.

    If constant is a scalar, the memory read is size of input and 1,
    Otherwise memory read is 2 times input size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input, or 1 + size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)


@register(Div)
def calc_div(func, in_data, **kwargs):
    """Div

    Elementwise Div operation.
    FLOPs is same as input element size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)


@register(DivFromConstant)
def calc_div_from_constant(func, in_data, **kwargs):
    """DivFromConstant

    Elementwise DivFrom operation with constant(s).
    FLOPs is same as input element size.

    If constant is a scalar, the memory read is size of input and 1,
    Otherwise memory read is 2 times input size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input, or 1 + size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)


@register(Mul)
def calc_mul(func, in_data, **kwargs):
    """Mul

    Elementwise Mul operation.
    FLOPs is same as input element size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)


@register(MulConstant)
def calc_mul_constant(func, in_data, **kwargs):
    """MulConstant

    Elementwise Mul operation with constant(s).
    FLOPs is same as input element size.

    If constant is a scalar, the memory read is size of input and 1,
    Otherwise memory read is 2 times input size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input, or 1 + size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)


@register(Sub)
def calc_sub(func, in_data, **kwargs):
    """Sub

    Elementwise Sub operation.
    FLOPs is same as input element size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)


@register(SubFromConstant)
def calc_sub_from_constant(func, in_data, **kwargs):
    """SubFromConstant

    Elementwise SubFrom operation with constant(s).
    FLOPs is same as input element size.

    If constant is a scalar, the memory read is size of input and 1,
    Otherwise memory read is 2 times input size.

    |Item|Value|
    |:---|:---|
    |FLOPs|size of input|
    |mread|2 times size of input, or 1 + size of input|
    |mwrite|size of input|
    """
    return _calc(func, in_data, **kwargs)
