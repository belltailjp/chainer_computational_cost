from chainer.functions.activation.relu import ReLU
from chainer.functions.activation.prelu import PReLUFunction
from chainer.functions.activation.sigmoid import Sigmoid


def calc_relu(func: ReLU, in_data, **kwargs):
    x, = in_data
    return (x.size, x.size, x.size)


def calc_sigmoid(func: Sigmoid, in_data, **kwargs):
    x, = in_data
    return (x.size, x.size, x.size)


def calc_prelu(func: PReLUFunction, in_data, **kwargs):
    x, W = in_data
    return (x.size, x.size + W.size, x.size)
