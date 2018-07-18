import chainer.functions as F
import numpy as np

from chainer_computational_cost.cost_calculators import calculators


def test_activation_prelu():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    W = np.random.randn(3).astype(np.float32)
    f = F.activation.prelu.PReLUFunction()
    flops, mread, mwrite, params = calculators[type(f)](f, [x, W])
    assert flops == x.size
    assert mread == x.size + W.size
    assert mwrite == x.size
    assert params == {'w_shape': W.shape}


def test_activation_relu():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.ReLU()
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert flops == x.size
    assert mread == x.size
    assert mwrite == x.size
    assert params == dict()


def test_activation_sigmoid():
    x = np.random.randn(1, 3, 100, 100).astype(np.float32)
    f = F.Sigmoid()
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    assert flops == x.size
    assert mread == x.size
    assert mwrite == x.size
    assert params == dict()


def test_softmax():
    x = np.random.randn(1, 100).astype(np.float32)
    f = F.activation.softmax.Softmax()
    flops, mread, mwrite, params = calculators[type(f)](f, [x])
    # flops: exp term, sum term, div term
    assert flops == x.size + (x.size - 1) + x.size
    assert mread == x.size
    assert mwrite == x.size
    assert params == dict()
