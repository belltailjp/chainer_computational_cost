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
        output_size = max(x.size for x in xs)
        flops = output_size * (len(xs) - 1)
        mread = sum(x.size for x in xs)
        return (flops, mread, output_size, {})


@register(Add)
def calc_add(func, in_data, **kwargs):
    """[Add](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    Elementwise Add operation.
    In case shape of inputs are different, broadcast will run and then
    elementwise arithmetic is conducted.
    Cost for broadcasting is ignored.
    For simplicity, it assumes all the arrays are first broadcasted to the
    output size then elementwise sum is calculated.
    The output size is the largest size of the input.

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ (N-1) \max_{i}^{N} \| x_{i} \| $$ |
    | mread  | $$ \sum_{i}^{N} \| x_{i} \| $$ |
    | mwrite | $$ \max_{i}^{N} \| x_{i} \| $$ |
    | params | N/A |
    """
    return _calc(func, in_data, **kwargs)


@register(AddConstant)
def calc_add_constant(func, in_data, **kwargs):
    """[AddConstant](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    AddConstant is elementwise Add operation where the operand is a constant
    (not a chainer.Variable but a numpy.array or a cupy.array).

    In case shape of inputs are different, broadcast will run and then
    elementwise arithmetic is conducted.  Cost for broadcasting is ignored.
    The output size is same as the larger one of either the input ($x$) or the
    operand (`$c$`).

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | $$ \max(\| x \|, \| c \|) \| $$ |
    | mread  | $$ \| x \| + \| c \| $$ |
    | mwrite | $$ \max(\| x \|, \| c \|) \| $$ |
    | params | N/A |
    """
    return _calc(func, in_data, **kwargs)


@register(Div)
def calc_div(func, in_data, **kwargs):
    """[Div](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    See the documentation for [Add](#add)
    """
    return _calc(func, in_data, **kwargs)


@register(DivFromConstant)
def calc_div_from_constant(func, in_data, **kwargs):
    """[DivFromConstant](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    See the documentation for [AddConstant](#addconstant)
    """
    return _calc(func, in_data, **kwargs)


@register(Mul)
def calc_mul(func, in_data, **kwargs):
    """[Mul](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    See the documentation for [Add](#add)
    """
    return _calc(func, in_data, **kwargs)


@register(MulConstant)
def calc_mul_constant(func, in_data, **kwargs):
    """[MulConstant](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    See the documentation for [AddConstant](#addconstant)
    """
    return _calc(func, in_data, **kwargs)


@register(Sub)
def calc_sub(func, in_data, **kwargs):
    """[Sub](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    See the documentation for [Add](#add)
    """
    return _calc(func, in_data, **kwargs)


@register(SubFromConstant)
def calc_sub_from_constant(func, in_data, **kwargs):
    """[SubFromConstant](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.add.html)

    See the documentation for [AddConstant](#addconstant)
    """
    return _calc(func, in_data, **kwargs)
