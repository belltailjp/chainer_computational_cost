from chainer_computational_cost.cost_calculators import register

from chainer.functions.activation.prelu import PReLUFunction
from chainer.functions.activation.relu import ReLU
from chainer.functions.activation.sigmoid import Sigmoid
from chainer.functions.activation.softmax import Softmax


@register(PReLUFunction)
def calc_prelu(func, in_data, **kwargs):
    x, W = in_data
    return (x.size, x.size + W.size, x.size, {})


@register(ReLU)
def calc_relu(func, in_data, **kwargs):
    x, = in_data
    return (x.size, x.size, x.size, {})


@register(Sigmoid)
def calc_sigmoid(func, in_data, **kwargs):
    x, = in_data
    return (x.size, x.size, x.size, {})


@register(Softmax)
def calc_softmax(func, in_data, **kwargs):
    x, = in_data
    return (x.size + (x.size - 1) + x.size, x.size, x.size, {})
