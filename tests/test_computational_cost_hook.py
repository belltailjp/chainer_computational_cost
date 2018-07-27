from collections import OrderedDict
import copy
import io
import sys

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

    # check summary report
    assert cost.summary_report['Convolution2DFunction']['n_layers'] == 3
    assert cost.summary_report['FixedBatchNormalization']['n_layers'] == 3
    assert cost.summary_report['LinearFunction']['n_layers'] == 2


def test_repeat():
    # To check if show_table doesn't break internal states
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()

    with chainer.using_config('train', False):
        with chainer_computational_cost.ComputationalCostHook() as cost:
            net(x)
            layer_report = copy.deepcopy(cost.layer_report)
            for mode in ['md', 'csv', 'table']:
                cost.show_report(mode=mode)
                cost.show_summary_report(mode=mode)
            assert cost.layer_report == layer_report


def test_custom_cost_calculator():
    called = {'called': False}

    def calc_custom(func, in_data, **kwargs):
        called['called'] = True  # should use nonlocal but py2 doesn't have it
        return (100, 100, 100, {})

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = chainer.Variable(x)
    with chainer.using_config('train', False):
        with chainer_computational_cost.ComputationalCostHook() as cost:
            with pytest.warns(UserWarning):
                cost.add_custom_cost_calculator(AddConstant, calc_custom)
                x = x + 1
                report = cost.layer_report

    report = report['AddConstant-1']
    assert called['called'] is True
    assert report['flops'] == 100
    assert report['mread'] == 100 * x.dtype.itemsize
    assert report['mwrite'] == 100 * x.dtype.itemsize
    assert report['mrw'] == report['mread'] + report['mwrite']


def test_custom_cost_calculator_invalid():
    def calc_no_type_annotation(func, in_data, **kwargs):
        pass

    def calc_not_tuple(func, in_data, **kwargs):
        return [1, 1, 1, dict()]

    def calc_insufficient_return(func, in_data, **kwargs):
        return (1, 1, 1)

    def calc_wrong_type(func, in_data, **kwargs):
        return (1, 1, 1, None)

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = chainer.Variable(x)
    for f in [calc_not_tuple, calc_not_tuple, calc_insufficient_return]:
        with chainer.using_config('train', False):
            with chainer_computational_cost.ComputationalCostHook() as cost:
                with pytest.raises(TypeError), pytest.warns(UserWarning):
                    cost.add_custom_cost_calculator(AddConstant, f)
                    x = x + 1


def test_report_ignored_layer():
    class DummyFunc(chainer.function_node.FunctionNode):

        def forward(self, xs):
            return xs

        def backward(self, indices, gys):
            return gys

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    with chainer_computational_cost.ComputationalCostHook() as cost:
        with pytest.warns(UserWarning):
            DummyFunc().apply(x)
            assert len(cost.ignored_layers) == 1
            assert 'DummyFunc' in list(cost.ignored_layers.keys())[0]
            assert 'DummyFunc' == list(cost.ignored_layers.values())[0]['type']


def test_show_report_unit_and_digits():
    conv = L.Convolution2D(32, 64, ksize=3, pad=1)
    x = np.random.randn(8, 32, 128, 128).astype(np.float32)
    with chainer_computational_cost.ComputationalCostHook() as ccost:
        conv(x)

    # Just check if assumed value are calculated
    assert len(ccost.summary_report) == 2   # Conv and Total
    assert ccost.summary_report['total']['flops'] == 2415919104
    assert ccost.summary_report['total']['mread'] == 16851200
    assert ccost.summary_report['total']['mwrite'] == 33554432
    assert ccost.summary_report['total']['mrw'] == 50405632

    def _report(report_func, **kwargs):
        # py2 support: csv and md mode doesn't support StringIO
        # cf. https://stackoverflow.com/questions/13120127
        if kwargs.get('mode') == 'table' or (3, 0) <= sys.version_info:
            sio = io.StringIO()
        else:
            sio = io.BytesIO()
        report_func(sio, **kwargs)
        s = sio.getvalue()

        # recognize table type and break down to a list of lists
        if ':---' in s:     # it's a markdown table
            return [[v for v in l.split('|') if len(v) != 0]
                    for l in s.splitlines()]
        elif ',' in s:      # csv
            return [l.split(',') for l in s.splitlines()]
        else:               # texttable
            return [[v.strip() for v in l.split('|') if len(v) != 0]
                    for l in s.splitlines()
                    if '---' not in l and '===' not in l]

    def show_report(**kwargs):
        return _report(ccost.show_report, **kwargs)

    def show_summary_report(**kwargs):
        return _report(ccost.show_summary_report, **kwargs)

    def assert_table(rep, expected):
        for col, val in expected.items():
            assert rep[col] == val

    col_flops, col_mr, col_mw, col_mrw = (1, 2, 3, 4)

    # Case unit=None: raw values are shown
    expected = {col_flops: '2415919104', col_mr: '16851200',
                col_mw: '33554432', col_mrw: '50405632'}
    assert_table(show_report(unit=None)[-1], expected)  # default CSV
    assert_table(show_report(unit=None, mode='md')[-1], expected)
    assert_table(show_report(unit=None, mode='table')[-1], expected)

    assert_table(show_report(unit=None)[-2], expected)

    # Case unit=G: FLOPs/=1000^3, mem/=1024^3, 3 digits after the decimal point
    expected = {col_flops: '2.416', col_mr: '0.016',
                col_mw: '0.031', col_mrw: '0.047'}
    assert_table(show_report(unit='G')[-1], expected)
    assert_table(show_report(unit='G', mode='md')[-1], expected)
    assert_table(show_report(unit='G', mode='table')[-1], expected)

    # Case unit=G, n_digits=6: more digits will be shown
    expected = {col_flops: '2.415919', col_mr: '0.015694',
                col_mw: '0.03125', col_mrw: '0.046944'}
    assert_table(show_report(unit='G', n_digits=6)[-1], expected)
    assert_table(show_report(unit='G', n_digits=6, mode='md')[-1], expected)
    assert_table(show_report(unit='G', n_digits=6, mode='table')[-1], expected)

    # Case unit=M, n_digits=0: Values are rounded to integer
    expected = {col_flops: '2416', col_mr: '16',
                col_mw: '32', col_mrw: '48'}
    assert_table(show_report(unit='M', n_digits=0)[-1], expected)
    assert_table(show_report(unit='M', n_digits=0, mode='md')[-1], expected)
    assert_table(show_report(unit='M', n_digits=0, mode='table')[-1], expected)

    # Case only some columns are specified
    rep = show_report(unit='G', columns=['name', 'mrw'])[-1]
    assert len(rep) == 2
    assert rep[1] == '0.047'    # only the specified column is reported

    # Case only some columns are specified (order is changed)
    rep = show_report(unit='G', columns=['name', 'mrw', 'flops'])[-1]
    assert len(rep) == 3
    assert rep[1] == '0.047'
    assert rep[2] == '2.416'    # flops comes after

    # Case when invalid column is specified
    with pytest.raises(ValueError):
        show_report(unit='G', columns=['name', 'wooooohoooooooo'])
