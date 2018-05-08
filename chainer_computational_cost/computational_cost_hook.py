from __future__ import print_function

from collections import OrderedDict
import inspect
import sys
import traceback

import chainer
from chainer_computational_cost.cost_calculators import calculators, _check_sig


class ComputationalCostHook(chainer.FunctionHook):
    _coeff_table = {
        None: 1, 'k': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12
    }

    _custom_cost_calculators = dict()

    def __init__(self, unify_fma=True):
        self._unify_fma = unify_fma
        self._label_count = dict()

        self.report = OrderedDict([
            ['total', {'ops': 0, 'mread': 0, 'mwrite': 0}]
        ])

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

    def forward_postprocess(self, function, in_data):
        if type(function) is chainer.function.FunctionAdapter:
            function = function._function
        label = type(function).__name__

        if type(function) in self._custom_cost_calculators:
            cal = self._custom_cost_calculators[type(function)]
        elif type(function) in calculators:
            cal = calculators[type(function)]
        else:
            print("Warning: {} is not yet supported by "
                  "ComputationalCostHook, ignored".format(label))
            return

        res = cal(function, in_data, unify_fma=self._unify_fma)
        ops, mread, mwrite = res

        # to bytes
        itemsize = in_data[0].dtype.itemsize
        mread *= itemsize
        mwrite *= itemsize

        # get stack trace
        tb = traceback.extract_stack()
        tb = tb[:-2]   # ignore first 2 items; extract_stack and this hook
        tb = traceback.format_list(tb)
        tb = ''.join(tb)

        if label not in self._label_count:
            self._label_count[label] = 0
        self._label_count[label] += 1

        name = '{}-{}'.format(label, self._label_count[label])
        self.report[name] = {
            'type': label,
            'ops': ops,
            'mread': mread,
            'mwrite': mwrite,
            'traceback': tb.strip()
        }
        report_total = self.report['total']
        report_total['ops'] += ops
        report_total['mread'] += mread
        report_total['mwrite'] += mwrite

    def show_report(self, dst=sys.stdout, mode='csv', unit='G'):
        if unit not in self._coeff_table:
            raise ValueError("Please specify either None, 'k', 'M', 'G' or 'T'"
                             " to argument `unit`.")
        coeff = self._coeff_table[unit]
        if unit is None:
            unit = ''

        # bring "total" to the tail
        total = self.report.pop('total')
        self.report['total'] = total

        if mode == 'csv':
            self._show_csv(dst, unit, coeff)
        if mode == 'md':
            self._show_md(dst, unit, coeff)
        elif mode == 'table':
            self._show_table(dst, unit, coeff)
        else:
            raise ValueError("Please specify either 'table' or 'md' to"
                             " argument `mode`")

    def _show_csv(self, ost, unit, coeff):
        ost.write("layer,{0}OPS,mread({0}B),mwrite({0}B)\n".format(unit))
        for layer, rep in self.report.items():
            ops, mread, mwrite = rep['ops'], rep['mread'], rep['mwrite']
            ops /= coeff
            mread /= coeff
            mwrite /= coeff
            ost.write("{},{},{},{}\n".format(layer, ops, mread, mwrite))

    def _show_md(self, ost, unit, coeff):
        ost.write("|layer|{0}OPS|mread({0}B)|mwrite({0}B)|\n".format(unit))
        ost.write("|:----|:----|:----|:----|\n")
        for layer, rep in self.report.items():
            ops, mread, mwrite = rep['ops'], rep['mread'], rep['mwrite']
            ops /= coeff
            mread /= coeff
            mwrite /= coeff
            ost.write("|{}|{}|{}|{}|\n".format(layer, ops, mread, mwrite))

    def _show_table(self, ost, unit, coeff):
        import texttable
        table = texttable.Texttable()

        rows = [['layer', '{}OPS'.format(unit),
                 'mread({}B)'.format(unit), 'mwrite({}B)'.format(unit)]]
        for layer, rep in self.report.items():
            ops, mread, mwrite = rep['ops'], rep['mread'], rep['mwrite']
            if coeff != 1:
                ops /= coeff
                mread /= coeff
                mwrite /= coeff
            rows.append([layer, str(ops), str(mread), str(mwrite)])
        table.add_rows(rows)
        ost.write(table.draw() + '\n')
