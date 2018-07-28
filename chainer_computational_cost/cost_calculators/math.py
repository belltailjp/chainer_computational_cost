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
from chainer.functions.math.minmax import ArgMax
from chainer.functions.math.minmax import ArgMin
from chainer.functions.math.minmax import Max
from chainer.functions.math.minmax import Min
from chainer.functions.math.sum import Sum


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


def _calc_minmax(func, x, **kwargs):
    if func.axis is None:
        return (x.size - 1, x.size, 1, {'axis': None})
    else:
        current_size = x.size
        flops = 0
        axes = func.axis
        if type(axes) is not tuple:  # argmin/argmax have only 1 int value
            axes = (axes,)
        for axis in axes:
            d = x.shape[axis]
            current_size //= d
            flops += (d - 1) * current_size
        return (flops, x.size, current_size, {'axis': func.axis})


@register(Max)
def calc_max(func, in_data, **kwargs):
    """[Max](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.max.html)

    In case `axis` is `None`, it just calculates maximum value of a tensor,
    which costs simply $\|x\|-1$.

    Here, let's consider a 4-dimensional array whose shape is $(A, B, C, D)$.
    When `axis` is set to `(1, 2)`,
    * First it calculates max over the axis 1, which is $(B-1)ACD$ FLOPs,
      and this makes a tensor shaped $(A, C, D)$
    * Next, max over the original axis 2 is conducted in $(C-1)AD$ FLOPs,
      and a tensor $(A, D)$ is remained.
    Total FLOPs is just a sum of above, and the output is $(A, D)$.

    Therefore, FLOPs is calculated by the following algorithm.

    ```
    input: x, axes
    output: f (FLOPs), s (output size)
    s <- x.size
    f <- 0
    foreach i in axes
        d <- x.shape
        s <- s / d
        f <- f + (d-1)s
    ```

    | Item   | Value |
    |:-------|:------|
    | FLOPs  | See the above explanation |
    | mread  | $$ \| x \|  $$ |
    | mwrite | See the above explanation |
    | params | `axis` |
    """
    x, = in_data
    return _calc_minmax(func, x, **kwargs)


@register(Min)
def calc_min(func, in_data, **kwargs):
    """[Min](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.min.html)

    See the documentation for [Max](#max).
    """
    x, = in_data
    return _calc_minmax(func, x, **kwargs)


@register(ArgMax)
def calc_argmax(func, in_data, **kwargs):
    """[ArgMax](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.argmax.html)

    Theoretical cost of Argmax is exactly same as Min/Max, except that
    Argmax can receive only one axis.
    See the documentation for [Max](#max).
    """
    x, = in_data
    return _calc_minmax(func, x, **kwargs)


@register(ArgMin)
def calc_argmin(func, in_data, **kwargs):
    """[ArgMin](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.argmin.html)

    Theoretical cost of Argmin is exactly same as Min/Max, except that
    Argmax can receive only one axis.
    See the documentation for [Max](#max).
    """
    x, = in_data
    return _calc_minmax(func, x, **kwargs)


@register(Sum)
def calc_sum(func, in_data, **kwargs):
    """[Sum](https://docs.chainer.org/en/v4.3.0/reference/generated/chainer.functions.sum.html)

    Sum of an array among the specified axis(axes) also costs equivalently to
    max operation, since it just replaces $\max(a, b)$ by $a+b$.
    See the documentation for [Max](#max).
    """
    x, = in_data
    return _calc_minmax(func, x, **kwargs)
