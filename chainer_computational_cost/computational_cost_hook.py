from collections import OrderedDict
import copy
import inspect
import itertools
import sys
import traceback
import warnings

import chainer
from chainer_computational_cost.cost_calculators import _check_sig
from chainer_computational_cost.cost_calculators import calculators


class ComputationalCostHook(chainer.FunctionHook):
    """Calculate theoretical computational cost of neural networks.

    This is a chainer hook. Inside the scope of an instance of this class,
    every chainer function call is caught by this hook and it accumulates
    theoretical computational cost (in FLOPs) and memory transfer.

    Args:
        fma_1flop: Specify `True` when you want to treat FMA (ax+b) as one
            floating point operation (default=`True`). Otherwise it is 2.
    """
    _flops_coeff_table = {
        None: 1, 'K': 10**3, 'M': 10**6, 'G': 10**9, 'T': 10**12
    }
    _bytes_coeff_table = {
        None: 1, 'K': 2**10, 'M': 2**20, 'G': 2**30, 'T': 2**40
    }
    _col_header_table = {
        'type': 'Layer type',
        'n_layers': '# Layers',
        'name': 'Layer name',
        'flops': '{0}FLOPs',        # {0} is k, M, G, ...
        'mread': 'MemRead\n{1}B',   # {1} is ki, Mi, Gi, ...
        'mwrite': 'MemWrite\n{1}B',
        'mrw': 'MemR+W\n{1}B',
        'input_shapes': 'Input shapes',
        'output_shapes': 'Output shapes',
        'params': 'Function parameters'
    }
    _custom_cost_calculators = dict()

    def __init__(self, fma_1flop=True):
        self._fma_1flop = fma_1flop
        self._label_count = dict()

        self.layer_report = OrderedDict()
        self.summary_report = OrderedDict()
        self.ignored_layers = OrderedDict()

    def add_custom_cost_calculator(self, calculator):
        """Add custom cost calculator function.

        This is an interface to extend the hook object so that the hook can
        handle unsupported layers or user-defined custom layers
        (and overwrite existing calculator).

        Args:
            calculator: Python function object whose signature is
                `def custom_calculator(func: F.math.basic_math.Add,
                in_data, **kwargs)`. The first argument name should be `func`,
                and it has to have type hinting. chainer-computational-cost
                hook calls your custom cost calculator when the function object
                matches to the specified type annotation. The second argument
                is the data fed to the function in the computational graph.
                The last argument should be `**kwargs`, which can include flags
                specified to ComputationalCostHook constructor
                (e.g. `fma_1flop`).  You can overwrite existing cost calculator
                by your custom one.
        """
        p = inspect.signature(calculator).parameters
        if not _check_sig(p):
            raise TypeError("Invalid signature for custom calculator."
                            "")
        func_type = p['func'].annotation
        if func_type in calculators:
            warnings.warn("replacing default cost calculator for {}"
                          .format(func_type.__name__))
        if func_type in self._custom_cost_calculators:
            type_name = func_type.__name__
            old_func_name = self._custom_cost_calculators[func_type].__name__
            warnings.warn("replacing existing custom cost calculator "
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
        """Hook function called by chainer.

        During the hook object lifetime, every chainer function calls and their
        inputs are passed to this method. It looks up cost calculator table and
        call the calculator with function object and input data.
        The returned informations are stored in this hook object.
        If an unsupported function appears in computational graph, it simply
        ignores them.
        """
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
            warnings.warn("{} is not yet supported by "
                          "ComputationalCostHook, ignored".format(fqn))
            self.ignored_layers[name] = {
                'type': label,
                'traceback': self._get_stack_trace(),
                'input_shapes': input_shapes,
                'output_shapes': output_shapes
            }
            return

        res = cal(function, in_data, fma_1flop=self._fma_1flop)
        err_msg = "Cost calculator has to return a tuple whose length is "\
                  "exactly 4 (flops: int, mread: int, "\
                  "mwrite: int, params: dict)."
        if type(res) != tuple or len(res) != 4:
            raise TypeError(err_msg)

        flops, mread, mwrite, params = res
        if type(flops) != int or type(mread) != int or \
                type(mwrite) != int or type(params) != dict:
            raise TypeError(err_msg)

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
                    'type': label, 'name': label, 'n_layers': 0,
                    'flops': 0, 'mread': 0, 'mwrite': 0, 'mrw': 0
                }
            report = self.summary_report[label]
            report['flops'] += flops
            report['n_layers'] += 1
            report['mread'] += mread
            report['mwrite'] += mwrite
            report['mrw'] += mread + mwrite

    def _get_fqn(self, func_type):
        return "{}.{}".format(func_type.__module__, func_type.__name__)

    def show_summary_report(self, ost=sys.stdout, mode='csv', unit='G',
                            columns=['type', 'n_layers', 'flops', 'mread',
                                     'mwrite', 'mrw']):
        """Show computational cost aggregated for each layer type.

        Summarizes based on chainer function. Every call of same function
        (e.g. F.convolution_2d) are aggregated and shown as a row.
        The output is sorted by the order each layer is called first.

        Args:
            ost: Output destination. It has to be a stream, by default
                `sys.stdout`.
            mode: `csv` (default), `md` and `table` are supported. When you use
                `table` mode, it requires texttable package.
            unit: Supplementary units used for both computational cost (FLOPs)
                and memory transfer (bytes). None, `K`, `M`, `G` (default) and
                `T` are supported.
            columns: which columns to include in summary.
        """
        # bring 'total' to the last
        report = copy.deepcopy(self.summary_report)
        report['total'] = report.pop('total')
        self._show_report_body(report, True, ost, mode, unit, columns)

    def show_report(self, ost=sys.stdout, mode='csv', unit='G',
                    columns=['name', 'flops', 'mread', 'mwrite', 'mrw']):
        """Show computational cost aggregated for each layer.

        Every single call of a function will appear as a row.
        The output is sorted by the order each layer is called.

        Args:
            ost: Output destination. It has to be a stream, by default
                `sys.stdout`.
            mode: `csv` (default), `md` and `table` are supported. When you use
                `table` mode, it requires texttable package.
            unit: Supplementary units used for both computational cost (FLOPs)
                and memory transfer (bytes). None, `K`, `M`, `G` (default) and
                `T` are supported.
            columns: which columns to include in summary.
        """
        # add 'total' to the last
        total = {'total': self.summary_report['total']}
        report = itertools.chain(self.layer_report.items(), total.items())
        report = OrderedDict(report)
        report = copy.deepcopy(report)
        self._show_report_body(report, False, ost, mode, unit, columns)

    def _show_report_body(self, report, summary, ost, mode, unit, cols):
        # check cols
        rep = list(report.values())[0]
        assert all([c in rep for c in cols]), \
            "Unknown column(s) specified: {}".format(cols)

        if unit not in self._flops_coeff_table:
            raise ValueError("Please specify either None, 'K', 'M', 'G' or 'T'"
                             " to argument `unit`.")
        coeff_flops = self._flops_coeff_table[unit]
        coeff_bytes = self._bytes_coeff_table[unit]
        if unit is None:
            unit = ''

        # make a header
        header = []
        for c in cols:
            # "{0}FLOPs" -> "GFLOPs", "{1}B/s" -> "GiB/s"
            fmt = self._col_header_table[c]
            fmt = fmt.format(unit, unit + 'i')
            header.append(fmt)

        # make table records
        table_report = [header]
        for layer, rep in report.items():
            if unit != '':
                rep['flops'] /= coeff_flops
                for c in ('mread', 'mwrite', 'mrw'):
                    rep[c] /= coeff_bytes
            if 'params' in rep:
                rep['params'] = self._prettify_dict(rep['params'])
            for c in cols:
                if c not in rep:
                    rep[c] = ''
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
            reps = ','.join([str(r) for r in reps])
            ost.write(reps.replace('\n', ' ') + '\n')

    def _show_md(self, table_report, ost):
        for i, reps in enumerate(table_report):
            if i == 1:
                ost.write('|:----' * len(reps) + '|\n')
            reps = '|'.join([str(r) for r in reps])
            ost.write('|' + reps.replace('\n', ' ') + '|\n')

    def _show_table(self, table_report, ost):
        import texttable
        table = texttable.Texttable(max_width=0)
        table.set_precision(6)
        table.add_rows(table_report)
        ost.write(table.draw() + '\n')
