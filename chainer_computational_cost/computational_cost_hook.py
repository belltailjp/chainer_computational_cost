from collections import OrderedDict
import inspect
import itertools
import sys
import traceback

import chainer
from chainer_computational_cost.cost_calculators import _check_sig
from chainer_computational_cost.cost_calculators import calculators


class ComputationalCostHook(chainer.FunctionHook):
    _coeff_table = {
        None: 1, 'k': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12
    }
    _col_header_table = {
        'type': 'Layer type',
        'name': 'Layer name',
        'flops': '{}FLOPs',
        'mread': 'MemRead\n{}B/s',
        'mwrite': 'MemWrite\n{}B/s',
        'mrw': 'MemR/W\n{}B/s',
        'input_shapes': 'Input shapes',
        'output_shapes': 'Output shapes',
        'params': 'Function parameters'
    }
    _custom_cost_calculators = dict()

    def __init__(self, unify_fma=True):
        self._unify_fma = unify_fma
        self._label_count = dict()

        self.layer_report = OrderedDict()
        self.summary_report = OrderedDict()
        self.ignored_layers = OrderedDict()

    def add_custom_cost_calculator(self, calculator):
        p = inspect.signature(calculator).parameters
        if not _check_sig(p):
            raise TypeError("Invalid signature for custom calculator."
                            "")
        func_type = p['func'].annotation
        if func_type in calculators:
            print("Warning: replacing default cost calculator for {}"
                  .format(func_type.__name__))
        if func_type in self._custom_cost_calculators:
            type_name = func_type.__name__
            old_func_name = self._custom_cost_calculators[func_type].__name__
            print("Warning: replacing existing custom cost calculator "
                  "for {} ({})".format(type_name, old_func_name))

        self._custom_cost_calculators[func_type] = calculator

    def _get_func_name_and_label(self, func_type):
        label = func_type.__name__
        if label not in self._label_count:
            self._label_count[label] = 0
        self._label_count[label] += 1

        name = '{}-{}'.format(label, self._label_count[label])
        return (label, name)

    def _get_stack_trace(self, ignore_depth=3):
        # ignore first 3 items;
        # extract_stack, forward_postprocess and _get_stack_trace
        tb = traceback.extract_stack()[:-ignore_depth]
        tb = traceback.format_list(tb)
        return ''.join(tb).strip()

    def forward_postprocess(self, function, in_data):
        if type(function) is chainer.function.FunctionAdapter:
            function = function._function

        outs = function.forward(in_data)
        input_shapes = [x.shape for x in in_data]
        output_shapes = [y.shape for y in outs]

        func_type = type(function)
        label, name = self._get_func_name_and_label(func_type)

        if func_type in self._custom_cost_calculators:
            cal = self._custom_cost_calculators[func_type]
        elif func_type in calculators:
            cal = calculators[func_type]
        else:
            fqn = self._get_fqn(func_type)
            print("Warning: {} is not yet supported by "
                  "ComputationalCostHook, ignored".format(fqn))
            self.ignored_layers[name] = {
                'type': label,
                'traceback': self._get_stack_trace(),
                'input_shapes': input_shapes,
                'output_shapes': output_shapes
            }
            return

        res = cal(function, in_data, unify_fma=self._unify_fma)
        flops, mread, mwrite, params = res

        # to bytes
        itemsize = in_data[0].dtype.itemsize
        mread *= itemsize
        mwrite *= itemsize

        self.layer_report[name] = {
            'name': name,
            'type': label,
            'flops': flops,
            'mread': mread,
            'mwrite': mwrite,
            'mrw': mread + mwrite,
            'traceback': self._get_stack_trace(),
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'params': params
        }

        for label in ('total', label):
            if label not in self.summary_report:
                self.summary_report[label] = {
                    'type': label, 'name': 'total', 'flops': 0,
                    'mread': 0, 'mwrite': 0, 'mrw': 0,
                    'input_shapes': '--', 'output_shapes': '--',
                    'params': {}
                }
            report = self.summary_report[label]
            report['flops'] += flops
            report['mread'] += mread
            report['mwrite'] += mwrite
            report['mrw'] += mread + mwrite

    def _get_fqn(self, func_type):
        return "{}.{}".format(func_type.__module__, func_type.__name__)

    def show_summary_report(self, ost=sys.stdout, mode='csv', unit='G',
                            columns=['type', 'flops', 'mread',
                                     'mwrite', 'mrw']):
        # bring 'total' to the last
        report = self.summary_report.copy()
        report['total'] = report.pop('total')
        self._show_report_body(report, True, ost, mode, unit, columns)

    def show_report(self, ost=sys.stdout, mode='csv', unit='G',
                    columns=['name', 'flops', 'mread', 'mwrite', 'mrw']):
        # add 'total' to the last
        total = {'total': self.summary_report['total']}
        report = itertools.chain(self.layer_report.items(), total.items())
        report = OrderedDict(report)
        self._show_report_body(report, False, ost, mode, unit, columns)

    def _show_report_body(self, report, summary, ost, mode, unit, cols):
        # check cols
        rep = list(report.values())[0]
        assert all([c in rep for c in cols]), \
            "Unknown column(s) specified: {}".format(cols)

        if unit not in self._coeff_table:
            raise ValueError("Please specify either None, 'k', 'M', 'G' or 'T'"
                             " to argument `unit`.")
        coeff = self._coeff_table[unit]
        if unit is None:
            unit = ''

        # make a header
        header = []
        for c in cols:
            fmt = self._col_header_table[c]
            if '{}' in fmt:
                fmt = fmt.format(unit)
            header.append(fmt)

        # make table records
        table_report = [header]
        for layer, rep in report.items():
            if unit != '':
                for c in ['flops', 'mread', 'mwrite', 'mrw']:
                    rep[c] /= coeff
            if 'params' in rep:
                rep['params'] = self._prettify_dict(rep['params'])
            table_report.append([rep[c] for c in cols])

        if mode == 'csv':
            self._show_csv(table_report, ost)
        elif mode == 'md':
            self._show_md(table_report, ost)
        elif mode == 'table':
            self._show_table(table_report, ost)
        else:
            raise ValueError("Please specify either 'table' or 'md' to"
                             " argument `mode`")

    def _prettify_dict(self, rep):
        return ', '.join(['{}={}'.format(k, v) for k, v in rep.items()])

    def _show_csv(self, table_report, ost):
        for reps in table_report:
            ost.write("{}\n".format(','.join([str(rep) for rep in reps])))

    def _show_md(self, table_report, ost):
        for i, reps in enumerate(table_report):
            if i == 1:
                ost.write('|:----' * len(reps) + '|\n')
            ost.write("|{}|\n".format('|'.join([str(r) for r in reps])))

    def _show_table(self, table_report, ost):
        import texttable
        table = texttable.Texttable(max_width=0)
        table.add_rows(table_report)
        ost.write(table.draw() + '\n')
