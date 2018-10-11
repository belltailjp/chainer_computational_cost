from collections import OrderedDict
import copy
import six

import chainer
import chainer.functions as F
from chainer.functions.math.basic_math import AddConstant
import chainer.links as L
import numpy as np

import pytest

from chainer_computational_cost import ComputationalCostHook


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
        with ComputationalCostHook() as cost:
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
        'LinearFunction-2',
        'total'
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
        with ComputationalCostHook() as cost:
            net(x)
            layer_report = copy.deepcopy(cost.layer_report)
            for mode in ['md', 'csv', 'table']:
                cost.show_report(mode=mode)
                cost.show_summary_report(mode=mode)
            assert cost.layer_report == layer_report


def test_report_property_keeps_internal_state():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost:
            net(x)

            for rep_name in ['layer_report', 'summary_report',
                             'total_report', 'ignored_layers']:
                # try to contaminate report
                rep = getattr(cost, rep_name)
                rep['hooo'] = 'hello'

                # but it does never break cost's internal state
                assert 'hooo' not in getattr(cost, rep_name)


def test_report_property_inserts_total_element():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost:
            net(x)
            total1 = cost.layer_report['total']
            total2 = cost.summary_report['total']
            assert total1 == total2
            assert total1['flops'] == cost.total_report['flops']
            assert total1['mread'] == cost.total_report['mread']
            assert total1['mwrite'] == cost.total_report['mwrite']
            assert total1['mrw'] == cost.total_report['mrw']
            assert total1['flops%'] == 100.0
            assert total1['mread%'] == 100.0
            assert total1['mwrite%'] == 100.0
            assert total1['mrw%'] == 100.0

            # but it doesn't break internal state
            assert 'total' not in cost._layer_report
            assert 'total' not in cost._summary_report


def test_report_property_inserts_percentage():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost:
            net(x)

            keys = 'flops%', 'mread%', 'mwrite%', 'mrw%'
            for rep in cost.layer_report.values():
                assert all(k in rep for k in keys)
            for rep in cost.summary_report.values():
                assert all(k in rep for k in keys)

            # but it doesn't break internal state
            for rep in cost._layer_report.values():
                assert all(k not in rep for k in keys)  # NOT!
            for rep in cost._layer_report.values():
                assert all(k not in rep for k in keys)  # NOT!


def test_custom_cost_calculator():
    called = {'called': False}

    def calc_custom(func, in_data, **kwargs):
        called['called'] = True  # should use nonlocal but py2 doesn't have it
        return (100, 100, 100, {})

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = chainer.Variable(x)
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost:
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
    calc_not_callable = 5

    def calc_invalid_signature1(a, b):
        pass

    def calc_invalid_signature2(a, b, c):
        pass

    def calc_no_type_annotation(func, in_data, **kwargs):
        pass

    def calc_not_tuple(func, in_data, **kwargs):
        return [1, 1, 1, dict()]

    def calc_insufficient_return(func, in_data, **kwargs):
        return (1, 1, 1)

    def calc_wrong_type(func, in_data, **kwargs):
        return (1, 1, 1, None)

    calculators = [
        calc_not_callable, calc_invalid_signature1, calc_invalid_signature2,
        calc_no_type_annotation, calc_not_tuple,
        calc_insufficient_return, calc_wrong_type
    ]

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x = chainer.Variable(x)
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost:
            for f in calculators:
                with pytest.raises(TypeError), pytest.warns(UserWarning):
                    cost.add_custom_cost_calculator(AddConstant, f)
                    x = x + 1
        with pytest.raises(TypeError):
            cost.add_custom_cost_calculator(1, f)


def test_report_ignored_layer():
    class DummyFunc(chainer.function_node.FunctionNode):

        def forward(self, xs):
            return xs

        def backward(self, indices, gys):
            return gys

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    with ComputationalCostHook() as cost:
        with pytest.warns(UserWarning):
            DummyFunc().apply(x)
            assert len(cost.ignored_layers) == 1
            assert 'DummyFunc' in list(cost.ignored_layers.keys())[0]
            assert 'DummyFunc' == list(cost.ignored_layers.values())[0]['type']


def test_blank_case():
    # Even if nothing happens while the hook lifetime,
    # it should not hang but show only a warning
    with ComputationalCostHook() as ccost:
        with pytest.warns(UserWarning):
            ccost.show_report()
        with pytest.warns(UserWarning):
            ccost.show_summary_report()


def test_show_report_unit_and_digits():
    conv = L.Convolution2D(32, 64, ksize=3, pad=1)
    x = np.random.randn(8, 32, 128, 128).astype(np.float32)
    with ComputationalCostHook() as ccost:
        conv(x)

    # Just check if assumed value are calculated
    assert len(ccost.summary_report) == 2   # Conv and Total
    assert ccost.summary_report['total']['flops'] == 2415919104
    assert ccost.summary_report['total']['mread'] == 16851200
    assert ccost.summary_report['total']['mwrite'] == 33554432
    assert ccost.summary_report['total']['mrw'] == 50405632

    def _report(report_func, **kwargs):
        sio = six.StringIO()
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

    def assert_table(rep, expect):
        for col, val in expect.items():
            assert rep[col] == val

    col_flops, col_mr, col_mw, col_mrw = (1, 2, 3, 4)

    # Case: default columns check
    report_cols = ['Layer name', 'GFLOPs',
                   'MemRead GiB', 'MemWrite GiB', 'MemR+W GiB']
    assert report_cols == show_report()[0]
    summary_cols = ['Layer type', '# Layers', 'GFLOPs',
                    'MemRead GiB', 'MemWrite GiB', 'MemR+W GiB']
    assert summary_cols == show_summary_report()[0]

    # Case unit=None: raw values are shown
    expect = {col_flops: '2415919104', col_mr: '16851200',
              col_mw: '33554432', col_mrw: '50405632'}
    assert_table(show_report(unit=None)[-1], expect)  # default CSV
    assert_table(show_report(unit=None, mode='md')[-1], expect)
    assert_table(show_report(unit=None, mode='table')[-1], expect)

    assert_table(show_report(unit=None)[-2], expect)

    # Case unit=G: FLOPs/=1000^3, mem/=1024^3, 3 digits after the decimal point
    expect = {col_flops: '2.416', col_mr: '0.016',
              col_mw: '0.031', col_mrw: '0.047'}
    assert_table(show_report(unit='G')[-1], expect)
    assert_table(show_report(unit='G', mode='md')[-1], expect)
    assert_table(show_report(unit='G', mode='table')[-1], expect)

    # Case unit=G, n_digits=6: more digits will be shown
    expect = {col_flops: '2.415919', col_mr: '0.015694',
              col_mw: '0.03125', col_mrw: '0.046944'}
    assert_table(show_report(unit='G', n_digits=6)[-1], expect)
    assert_table(show_report(unit='G', n_digits=6, mode='md')[-1], expect)
    assert_table(show_report(unit='G', n_digits=6, mode='table')[-1], expect)

    # Case unit=G, n_digits>10: truncated to 10 digits
    expect = {col_flops: '2.415919104', col_mr: '0.015693903',
              col_mw: '0.03125', col_mrw: '0.046943903'}
    assert_table(show_report(unit='G', n_digits=11)[-1], expect)
    assert_table(show_report(unit='G', n_digits=11, mode='md')[-1], expect)
    assert_table(show_report(unit='G', n_digits=11, mode='table')[-1], expect)

    # Case unit=M, n_digits=0: Values are rounded to integer
    expect = {col_flops: '2416', col_mr: '16',
              col_mw: '32', col_mrw: '48'}
    assert_table(show_report(unit='M', n_digits=0)[-1], expect)
    assert_table(show_report(unit='M', n_digits=0, mode='md')[-1], expect)
    assert_table(show_report(unit='M', n_digits=0, mode='table')[-1], expect)

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


def test_error_when_invalid_report_type_is_specified():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    net = SimpleConvNet()
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost:
            net(x)
    with pytest.raises(ValueError):
        cost.show_report(mode='unknown')
    with pytest.raises(ValueError):
        cost.show_report(unit='unknown')
    with pytest.raises(ValueError):
        cost.show_report(n_digits=-1)


def test_nest():
    x = chainer.Variable(np.zeros((1, 3, 32, 32)).astype(np.float32))
    c = chainer.Variable(np.ones((1, 3, 32, 32)).astype(np.float32))
    with chainer.using_config('train', False):
        with ComputationalCostHook() as cost1:
            x = x + c
            with ComputationalCostHook() as cost2_1:
                x = x + c
                assert cost2_1.name == 'ComputationalCostHook-2'
                assert cost2_1.layer_report['total']['flops'] == 3 * 32 * 32

            with ComputationalCostHook() as cost2_2:
                x = x + c
                assert cost2_2.name == 'ComputationalCostHook-2'
                assert cost2_2.layer_report['total']['flops'] == 3 * 32 * 32

            assert cost1.name == 'ComputationalCostHook-1'
            assert cost1.layer_report['total']['flops'] == 3 * 3 * 32 * 32
