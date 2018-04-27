from __future__ import print_function

from collections import OrderedDict

import chainer
from . import cost_calculators


class ComputationalCostHook(chainer.FunctionHook):
    _coeff_table = {
        None: 1, 'k': 10**3, 'M': 10**6, 'G': 10**6, 'T': 10**9
    }

    def __init__(self, unify_fma=True):
        self._unify_fma = unify_fma
        self._label_count = dict()

        self.report = OrderedDict([
            ['total', {'ops': 0, 'mread': 0, 'mwrite': 0}]
        ])

    def forward_postprocess(self, function, in_data):
        label = function.label
        if label in cost_calculators.calculators:
            cal = cost_calculators.calculators[label]
            res = cal(function, in_data, unify_fma=self._unify_fma)
            ops, mread, mwrite = res

            # to bytes
            itemsize = in_data[0].dtype.itemsize
            mread *= itemsize
            mwrite *= itemsize

            if label not in self._label_count:
                self._label_count[label] = 0

            name = '{}-{}'.format(label, self._label_count[label])
            self._label_count[label] += 1
            self.report[name] = {
                'ops': ops,
                'mread': mread,
                'mwrite': mwrite
            }
            report_total = self.report['total']
            report_total['ops'] += ops
            report_total['mread'] += mread
            report_total['mwrite'] += mwrite

        else:
            print("Warning: {} is not yet supported by "
                  "ComputationalCostHook, ignored".format(label))

    def show_report(self, ost, mode='csv', unit='G'):
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
            self._show_csv(ost, unit, coeff)
        if mode == 'md':
            self._show_md(ost, unit, coeff)
        elif mode == 'table':
            self._show_table(ost, unit, coeff)
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
            ost.write("{},{},{}\n".format(layer, ops, mread, mwrite))

    def _show_md(self, ost, unit, coeff):
        ost.write("|layer|{0}OPS|mread({0}B)|mwrite({0}B)|\n".format(unit))
        ost.write("|:----|:----|:----|:----|\n")
        for layer, rep in self.report.items():
            ops, mread, mwrite = rep['ops'], rep['mread'], rep['mwrite']
            ops /= coeff
            mread /= coeff
            mwrite /= coeff
            ost.write("|{}|{}|{}|\n".format(layer, ops, mread, mwrite))

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

