from collections import OrderedDict

import chainer
import chainer.functions as F
from chainer.functions.math.basic_math import AddConstant
import chainer.links as L
import numpy as np

import pytest

import chainer_computational_cost


class SimpleConvNet(chainer.Chain):
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.bn1 = L.BatchNormalization(32)
            self.conv2 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.bn2 = L.BatchNormalization(32)
            self.conv3 = L.Convolution2D(None, 32, ksize=3, stride=1, pad=1)
            self.bn3 = L.BatchNormalization(32)
            self.fc4 = L.Linear(None, 100)
            self.fc5 = L.Linear(None, 10)

    def __call__(self, h):
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = F.relu(self.bn3(self.conv3(h)))
        height, width = h.shape[2:]
        h = F.average_pooling_2d(h, ksize=(height, width))
        h = F.reshape(h, (h.shape[0], -1))
        h = F.relu(self.fc4(h))
        return self.fc5(h)


def test_simple_net():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()
    with chainer.using_config('train', False):
        with chainer_computational_cost.ComputationalCostHook() as cost:
            net(x)
            assert type(cost.layer_report) == OrderedDict

    # check report existence and order
    reports = cost.layer_report
    assert list(reports.keys()) == [
        'Convolution2DFunction-1',
        'FixedBatchNormalization-1',
        'ReLU-1',
        'Convolution2DFunction-2',
        'FixedBatchNormalization-2',
        'ReLU-2',
        'Convolution2DFunction-3',
        'FixedBatchNormalization-3',
        'ReLU-3',
        'AveragePooling2D-1',
        'Reshape-1',
        'LinearFunction-1',
        'ReLU-4',
        'LinearFunction-2'
    ]

    # check parameters are properly reported
    assert reports['Convolution2DFunction-1']['params']['k'] == 3
    assert reports['Convolution2DFunction-1']['params']['groups'] == 1
    assert reports['ReLU-1']['params'] == dict()
    assert reports['AveragePooling2D-1']['params']['k'] == 32

    # check input and output shapes are properly reported
    conv_report = reports['Convolution2DFunction-1']
    assert len(conv_report['input_shapes']) == 3
    assert conv_report['input_shapes'][0] == (1, 3, 32, 32)
    assert conv_report['input_shapes'][1] == (32, 3, 3, 3)
    assert conv_report['input_shapes'][2] == (32,)
    assert len(conv_report['output_shapes']) == 1
    assert conv_report['output_shapes'][0] == (1, 32, 32, 32)


def test_custom_cost_calculator():
    called = False

    def calc_custom(func: AddConstant, in_data, **kwargs):
        nonlocal called
        called = True
        return (100, 100, 100, {})

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = chainer.Variable(x)
    with chainer.using_config('train', False):
        with chainer_computational_cost.ComputationalCostHook() as cost:
            cost.add_custom_cost_calculator(calc_custom)
            x = x + 1
            report = cost.layer_report

    report = report['AddConstant-1']
    assert called is True
    assert report['flops'] == 100
    assert report['mread'] == 100 * x.dtype.itemsize
    assert report['mwrite'] == 100 * x.dtype.itemsize
    assert report['mrw'] == report['mread'] + report['mwrite']


def test_custom_cost_calculator_invalid():
    def calc_invalid_custom(func, in_data, **kwargs):
        pass

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = chainer.Variable(x)
    with chainer.using_config('train', False):
        with chainer_computational_cost.ComputationalCostHook() as cost:
            with pytest.raises(TypeError):
                cost.add_custom_cost_calculator(calc_invalid_custom)
                x = x + 1


def test_report_ignored_layer():
    class DummyFunc(chainer.function_node.FunctionNode):

        def forward(self, xs):
            return xs

        def backward(self, indices, gys):
            return gys

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    with chainer_computational_cost.ComputationalCostHook() as cost:
        DummyFunc().apply(x)
        assert len(cost.ignored_layers) == 1
        assert 'DummyFunc' in list(cost.ignored_layers.keys())[0]
        assert 'DummyFunc' == list(cost.ignored_layers.values())[0]['type']
