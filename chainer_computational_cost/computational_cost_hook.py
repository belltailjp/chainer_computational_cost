from collections import OrderedDict
import copy
import sys
import traceback
import warnings

import chainer
from chainer_computational_cost.cost_calculators import calculators
from chainer_computational_cost.cost_calculators import check_signature


class ReportColumns(object):
    """Predefined column definitions used for ComputationalCostHook.show_report

    User should specify an array of column names to report to the `columns`
    argument of `ComputationalCostHook.show_report` method.
    This class provides some predefined sets. `DEFAULT` is set as a
    default. You can either specify like
    `cost.show_report(columns=ReportColumns.ALL)` or manually pass a list of
    column name that you exactly want to look.

    * DEFAULT: default columns
    * DEFAULT_AND_PERCENT: in addition to defaults, percentage columns are
      shown
    * ALL: all the possible columns, including input/output shape and
      supplemental informations for each layer
    """
    DEFAULT = ['name', 'flops', 'mread', 'mwrite', 'mrw']
    DEFAULT_AND_PERCENT = ['name', 'flops', 'mread', 'mwrite', 'mrw',
                           'flops%', 'mread%', 'mwrite%', 'mrw%']
    ALL = ['name', 'flops', 'mread', 'mwrite', 'mrw',
           'flops%', 'mread%', 'mwrite%', 'mrw%',
           'input_shapes', 'output_shapes', 'params']


class SummaryColumns(object):
    """Predefined column definitions used for ComputationalCostHook.show_summary_report

    Similar to `ReportColumns`, this is used for the `columns` argument of
    `ComputationalCostHook.show_summary_report` method.

    * DEFAULT: default columns
    * ALL: in addition to defaults, percentage columns are shown
    """     # NOQA
    DEFAULT = ['type', 'n_layers', 'flops', 'mread', 'mwrite', 'mrw']
    ALL = ['type', 'n_layers', 'flops', 'mread', 'mwrite', 'mrw',
           'flops%', 'mread%', 'mwrite%', 'mrw%']


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
        'flops%': 'FLOPs\n(%)',
        'mread%': 'MemRead\n(%)',
        'mwrite%': 'MemWrite\n(%)',
        'mrw%': 'MemR+W\n(%)',
        'input_shapes': 'Input shapes',
        'output_shapes': 'Output shapes',
        'params': 'Function parameters'
    }
    _custom_cost_calculators = dict()
    max_digits = 10

    def __init__(self, fma_1flop=True):
        self._fma_1flop = fma_1flop
        self._label_count = dict()

        self._layer_report = OrderedDict()
        self._summary_report = OrderedDict()
        self._ignored_layers = OrderedDict()
        self._total_report = {
            'name': 'total', 'type': 'total'
        }

    def add_custom_cost_calculator(self, func_type, calculator):
        """Add custom cost calculator function.

        This is an interface to extend the hook object so that the hook can
        handle unsupported layers, user-defined custom layers, or overwrite
        behavior of default cost calculator that chainer-computational-cost
        provides.
        After registering a pair of a Chainer(-compatible) function type and a
        custom cost calculator function to a ComputationalCostHook object,
        it calls the registered calculator if a specified type of chainer
        function is called inside a computational graph.

        Args:
            func_type: Type object of Chainer function
                (for example `chainer.functions.activation.relu.ReLU`).
            calculator: Python function object whose signature is
                `def custom_calculator(func, in_data, **kwargs)`.

                It has to receive 3 arguments, where the first argument will
                receive Function or FunctionNode object in computational graph.
                The second one is the data fed to the function, which is a
                list of array (numpy.array or cupy.array).
                The last one is a keyword-argument, which can include flags
                specified to ComputationalCostHook constructor
                (e.g. `fma_1flop`).

                It has to return a 4-element tuple.
                The 1st element (int) is the amount of of floating point
                arithmetics that the layer theoretically conducts.
                The 2nd and 3rd are the theoretical memory transfer of the
                layer for read and write, respectively. Be careful that the
                unit is the number of elements in int, not bytes.
                The last element is a dict of parameters of the layer.
                For example, cost calculator for Conv2D returns value of `k`,
                `pad` and so on, which helps user to identify the layer.
                Any informations can be returned.

                You can overwrite existing cost calculators by your custom one,
                regardless of it is provided by chainer-computational-cost or
                a custom cost calculator. Only the last calculator registered
                to the ComputationalCostHook object will be called.
        """
        if not isinstance(func_type, type):
            raise TypeError("Invalid func_type is specified. "
                            "Please specify type object.")
        if not check_signature(calculator):
            raise TypeError("Invalid signature for custom calculator.")

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

    def _get_fqn(self, func_type):
        return "{}.{}".format(func_type.__module__, func_type.__name__)

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
            self._ignored_layers[name] = {
                'type': label,
                'traceback': self._get_stack_trace(),
                'input_shapes': input_shapes,
                'output_shapes': output_shapes
            }
            return

        res = cal(function, in_data, fma_1flop=self._fma_1flop)
        err_msg = "Cost calculator has to return a tuple whose length is "\
                  "exactly 4 (flops: int, mread: int, "\
                  "mwrite: int, params: dict). "
        if type(res) != tuple:
            raise TypeError(err_msg + "{} is specified.".format(type(res)))
        if len(res) != 4:
            raise TypeError(err_msg + "The specified length is {}"
                            .format(len(res)))

        flops, mread, mwrite, params = res
        if type(flops) != int or type(mread) != int or \
                type(mwrite) != int or type(params) != dict:
            msg2 = "({}, {}, {}, {}) is specified".format(
                   type(flops), type(mread), type(mwrite), type(params))
            raise TypeError(err_msg + msg2)

        # to bytes
        itemsize = in_data[0].dtype.itemsize
        mread *= itemsize
        mwrite *= itemsize

        self._layer_report[name] = {
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

        if label not in self._summary_report:
            self._summary_report[label] = {'type': label, 'name': label}

        # Make layer type wise summary and overall summary
        for report in (self._summary_report[label], self._total_report):
            report['flops'] = report.get('flops', 0) + flops
            report['n_layers'] = report.get('n_layers', 0) + 1
            report['mread'] = report.get('mread', 0) + mread
            report['mwrite'] = report.get('mwrite', 0) + mwrite
            report['mrw'] = report.get('mrw', 0) + mread + mwrite

    def _insert_percentage(self, records):
        if len(records) == 0:
            return OrderedDict()

        # Insert "total" at the bottom
        records = copy.deepcopy(records)
        records['total'] = self.total_report

        # Insert percentage columns
        total = self.total_report
        for record in records.values():
            for key in ('flops', 'mread', 'mwrite', 'mrw'):
                record[key + '%'] = 100.0 * record[key] / total[key]
        return records

    @property
    def layer_report(self):
        """Get computational cost estimation of all layers

        Layer-wise cost estimaion is returned as an OrderedDict.
        The record is sorted in the ascending order by the time each layer is
        called.

        Key of the dict is a name of a layer in `str`. This name is
        automatically determined by chainer-computational-cost.
        Value is a `dict` of estimatied computational costs and some
        supplemental information as follows
        * `type`: type of the layer (name of `Function` class)
        * `name`: layer name "(type)-(order)", unique in the hook object
          lieftime
        * `flops`, `mread`, `mwrite` and `mrw`: computational cost estimations
        * `flops%`, `mread%`, `mwrite%` and `mrw%`: percentage of estimations
        * `traceback`: where the layer is called in the source code
        * `input_shapes` and `output_shapes`

        The last item of the returned records is "total".

        Even if you change the content of the returned value,
        that will not affect to the internal state of the hook object.
        """
        return self._insert_percentage(self._layer_report)

    @property
    def summary_report(self):
        """Get layer type wise computational cost estimation

        Similar to `layer_report` but each cost is summarized for each layer
        type. It has the following elements.
        * `type` and `name`: type of the layer (same value)
        * `n_layers`: number of this type of layer in the network
        * `flops`, `mread`, `mwrite` and `mrw`: computational cost estimations
        * `flops%`, `mread%`, `mwrite%` and `mrw%`: percentage of estimations

        It also has the record "total" in the last.

        Even if you change the content of the returned value,
        that will not affect to the internal state of the hook object.
        """
        return self._insert_percentage(self._summary_report)

    @property
    def total_report(self):
        """Total computational cost caught during the hook lifetime

        It returns a `dict` with the following elements
        * `name` = `type` = `'total'`
        * `n_layers`: number of this type of layer in the network
        * `flops`, `mread`, `mwrite` and `mrw`: computational cost estimations

        Even if you change the content of the returned value,
        that will not affect to the internal state of the hook object.
        """
        return copy.deepcopy(self._total_report)

    @property
    def ignored_layers(self):
        """List of ignored layers

        When a layer whose type is not supported by chainer-computational-cost
        yet, it is ignored and recorded into `ignored_layers`.

        Even if you change the content of the returned value,
        that will not affect to the internal state of the hook object.
        """
        return copy.deepcopy(self._ignored_layers)

    def show_summary_report(self, ost=sys.stdout, mode='csv', unit='G',
                            columns=SummaryColumns.DEFAULT, n_digits=3):
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
            n_digits: Specify how many digits after the deciaml point to show.
                Default is 3. Minimum value is 0 where all the values are
                rounded to integer, and maximum is 10. When `None` is specified
                it does not round, so as much digits as possible will be shown,
                that is equivalent to maximum value 10.
                Be noted that the value specified for `n_digits` does not
                affect to the precision of the summary row in the bottom of the
                table. This is because summary is calculated *before* rounding.
        """
        if len(self.summary_report) == 0:
            warnings.warn("No chainer function is caught during "
                          "lifetime of the hook")
            return
        self._show_report_body(self.summary_report, ost, mode, unit,
                               columns, n_digits)

    def show_report(self, ost=sys.stdout, mode='csv', unit='G',
                    columns=ReportColumns.DEFAULT, n_digits=3):
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
            columns: Which columns to include in summary.
            n_digits: Specify how many digits after the deciaml point to show.
                Default is 3. Minimum value is 0 where all the values are
                rounded to integer, and maximum is 10. When `None` is specified
                it does not round, so as much digits as possible will be shown,
                that is equivalent to maximum value 10.
                Be noted that the value specified for `n_digits` does not
                affect to the precision of the summary row in the bottom of the
                table. This is because summary is calculated *before* rounding.
        """
        if len(self.layer_report) == 0:
            warnings.warn("No chainer function is caught during "
                          "lifetime of the hook")
            return
        self._show_report_body(self.layer_report, ost, mode, unit,
                               columns, n_digits)

    def _show_report_body(self, report, ost, mode, unit, cols,
                          n_digits):
        if n_digits is None or self.max_digits < n_digits:
            n_digits = self.max_digits
        if not isinstance(n_digits, int) or n_digits < 0:
            raise ValueError("n_digits must be either None or an integer "
                             "larger or equal to 0, but {} ({}) is specified"
                             .format(n_digits, type(n_digits)))

        if n_digits == 0:
            rounder = lambda t: str(int(round(t, 0)))
        else:
            rounder = lambda t: str(round(t, n_digits))

        # check cols
        rep = list(report.values())[0]
        if any([c not in rep for c in cols]):
            raise ValueError("Unknown column(s) specified: {}\n"
                             "Available options: {}"
                             .format(cols, ", ".join(rep.keys())))

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
            fmt = fmt.format(unit, unit + 'i' if len(unit) else '')
            header.append(fmt)

        # make table records
        table_report = [header]
        for layer, rep in report.items():
            # round estimations (and add prefixed unit)
            if unit != '':
                flops = rounder(float(rep['flops']) / coeff_flops)
                rep['flops'] = flops
                for c in ('mread', 'mwrite', 'mrw'):
                    rep[c] = rounder(float(rep[c]) / coeff_bytes)

            # round percentage field
            for c in ('flops%', 'mread%', 'mwrite%', 'mrw%'):
                rep[c] = rounder(rep[c]) + '%'

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
        table.set_cols_dtype(['t'] * len(table_report[0]))  # everything text
        table.add_rows(table_report)
        ost.write(table.draw() + '\n')
