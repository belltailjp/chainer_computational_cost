# chainer-computational-cost

[![Build Status](https://travis-ci.org/belltailjp/chainer_computational_cost.svg?branch=master)](https://travis-ci.org/belltailjp/chainer_computational_cost)
[![Coverage Status](https://coveralls.io/repos/github/belltailjp/chainer_computational_cost/badge.svg?branch=master)](https://coveralls.io/github/belltailjp/chainer_computational_cost?branch=master)

This is a tool to estimate theoretical computational cost
of a chainer-based neural network forward pass.

You can analyze

* Theoretical amount of floating point arithmetics (FLOPs)
* Theoretical amount of memory read and write (mread/mwrite)

For each layer (we call them _computational costs_).

Also, summary of these computational costs for each layer-type,
and total cost can be calculated.

The computational costs this tool estimates are all **theoretical** number,
by assuming a most straightforward naive implementation, **for each layer**.
Therefore, for example, the following factors are **NOT** considered.

* Layer fusion
  * It is general to fuse Conv-BN-ReLU stack into one layer,
    that reduces memory transfer significantly.
* Techniques for speeding-up convolution
  * Lowering (im2col)
  * Winograd

In addition, for now, they are not in cosideration either.

* Mixed precision network
  * For example, keeping conv weights in FP16, saves as FP32
  * chainer-computational-cost uses data type of input tensor of a layer

Otherwise, these costs are also exluceded from computational cost.

* Inside-layer working memory read/write
  * Assume that every layers read only the input data and parameters from memory
    and write only the final results back to memory
* Index calculation
  * For example, `shift` and `transpose` heavily relies on src/dst index
    calculation but these are ignored
* Broadcasting


However,

* Duplicated memory access are ignored
  * In the most naive implementation, for example ResizeImages and Conv will
    read memory from the same region repeatedly. chainer-computational-cost
    treats such cases as just reading the entire input only once


## Requirements

* Python 2.7, 3.3, 3.4, 3.5, 3.6 and 3.7
* Chainer 3.5, 4.4, 5.0.0rc1
* six==1.11.0
* (optional) texttable >= 1.4.0
* (optional) pytest >= 3.5.1


## Installation

```bash
% pip install chainer_computational_cost
```

Manual installation by

```bash
% git clone git@github.com:belltailjp/chainer_computational_cost.git
% cd chainer_computational_cost
% python setup.py
```

or

```bash
% pip install git+https://github.com/belltailjp/chainer_computational_cost.git
```


## Quick Start

```python
import chainer
import chainer.links as L
import numpy as np

from chainer_computational_cost import ComputationalCostHook

net = L.VGG16Layers()
x = np.random.random((1, 3, 224, 224)).astype(np.float32)
with chainer.no_backprop_mode(), chainer.using_config('train', False):
    with ComputationalCostHook(fma_1flop=True) as cch:
        y = net(x)
        cch.show_report(unit='G', mode='md')
```

It will show the following table to stdout in markdown table format.

|Layer name|GFLOPs|MemRead GiB|MemWrite GiB|MemR+W GiB|
|:----|:----|:----|:----|:----|
|Convolution2DFunction-1|0.087|0.001|0.012|0.013|
|ReLU-1|0.003|0.012|0.012|0.024|
|Convolution2DFunction-2|1.85|0.012|0.012|0.024|
|ReLU-2|0.003|0.012|0.012|0.024|
|MaxPooling2D-1|0.002|0.012|0.003|0.015|
|Convolution2DFunction-3|0.925|0.003|0.006|0.009|
|ReLU-3|0.002|0.006|0.006|0.012|
|Convolution2DFunction-4|1.85|0.007|0.006|0.013|
|ReLU-4|0.002|0.006|0.006|0.012|
|MaxPooling2D-2|0.001|0.006|0.001|0.007|
|Convolution2DFunction-5|0.925|0.003|0.003|0.006|
|ReLU-5|0.001|0.003|0.003|0.006|
|Convolution2DFunction-6|1.85|0.005|0.003|0.008|
|ReLU-6|0.001|0.003|0.003|0.006|
|Convolution2DFunction-7|1.85|0.005|0.003|0.008|
|ReLU-7|0.001|0.003|0.003|0.006|
|MaxPooling2D-3|0.001|0.003|0.001|0.004|
|Convolution2DFunction-8|0.925|0.005|0.001|0.007|
|ReLU-8|0.0|0.001|0.001|0.003|
|Convolution2DFunction-9|1.85|0.01|0.001|0.012|
|ReLU-9|0.0|0.001|0.001|0.003|
|Convolution2DFunction-10|1.85|0.01|0.001|0.012|
|ReLU-10|0.0|0.001|0.001|0.003|
|MaxPooling2D-4|0.0|0.001|0.0|0.002|
|Convolution2DFunction-11|0.462|0.009|0.0|0.01|
|ReLU-11|0.0|0.0|0.0|0.001|
|Convolution2DFunction-12|0.462|0.009|0.0|0.01|
|ReLU-12|0.0|0.0|0.0|0.001|
|Convolution2DFunction-13|0.462|0.009|0.0|0.01|
|ReLU-13|0.0|0.0|0.0|0.001|
|MaxPooling2D-5|0.0|0.0|0.0|0.0|
|Reshape-1|0.0|0.0|0.0|0.0|
|LinearFunction-1|0.103|0.383|0.0|0.383|
|ReLU-14|0.0|0.0|0.0|0.0|
|LinearFunction-2|0.017|0.063|0.0|0.063|
|ReLU-15|0.0|0.0|0.0|0.0|
|LinearFunction-3|0.004|0.015|0.0|0.015|
|Softmax-1|0.0|0.0|0.0|0.0|
|total|15.488|0.623|0.107|0.729|


If you call `show_summary_report` method,
it will show summary for each type of layer.

|Layer type|# Layers|GFLOPs|MemRead GiB|MemWrite GiB|MemR+W GiB|
|:----|:----|:----|:----|:----|:----|
|Convolution2DFunction|13|15.347|0.089|0.05|0.139|
|ReLU|15|0.014|0.05|0.05|0.101|
|MaxPooling2D|5|0.005|0.023|0.006|0.029|
|Reshape|1|0.0|0.0|0.0|0.0|
|LinearFunction|3|0.124|0.461|0.0|0.461|
|Softmax|1|0.0|0.0|0.0|0.0|
|total|38|15.488|0.623|0.107|0.729|


Estimation values can be accessed through instance method of
`ComputationalCostHook`.

* `layer_report`
  * Layer-wise computational-cost estimations
* `summary_report`
  * Computational costs summarized for each layer types
* `ignored_layers`
  * List of layers that are not yet supported by chainer-computational-cost
* `total_report`
  * Total computational costs of the entire NN
    (layers caught during the lifetime of the hook object)


## Usage

As for basic usage, please refer to the avobe quickstart.


### FMA mode

When `fma_1flop` is set to `True`, chainer_computational_cost considers
FMA (fused multiply and add, `ax + b`) as one operation.
Otherwise, it counts as 2 operations.

This affects to convolution and linear layers' estimation.


### Reporting

Estimated computational cost table is reported by calling `show_report`
and `show_summary_report` method.

These have several options as explained below.


#### Report mode

Currently it supports the following modes.

* CSV mode (`mode='csv'`) - default
* Markdown table (`mode='md'`)
* Prettified table (`mode='table'`)

```
>>> cch.show_summary_report(unit='G', mode='table')
+-----------------------+----------+--------+---------+----------+--------+
|      Layer type       | # Layers | GFLOPs | MemRead | MemWrite | MemR+W |
|                       |          |        |   GiB   |   GiB    |  GiB   |
+=======================+==========+========+=========+==========+========+
| Convolution2DFunction | 13       | 15.347 | 0.089   | 0.050    | 0.139  |
+-----------------------+----------+--------+---------+----------+--------+
| ReLU                  | 15       | 0.014  | 0.050   | 0.050    | 0.101  |
+-----------------------+----------+--------+---------+----------+--------+
| MaxPooling2D          | 5        | 0.005  | 0.023   | 0.006    | 0.029  |
+-----------------------+----------+--------+---------+----------+--------+
| Reshape               | 1        | 0      | 0       | 0        | 0      |
+-----------------------+----------+--------+---------+----------+--------+
| LinearFunction        | 3        | 0.124  | 0.461   | 0        | 0.461  |
+-----------------------+----------+--------+---------+----------+--------+
| Softmax               | 1        | 0      | 0       | 0        | 0      |
+-----------------------+----------+--------+---------+----------+--------+
| total                 | 38       | 15.488 | 0.623   | 0.107    | 0.729  |
+-----------------------+----------+--------+---------+----------+--------+
```


#### Report destination

Report is by default written to stdout.
You can specify stream (e.g. file object) to `dst` argument of
`show_report` and `show_summary_report`.

```python
cch.show_report(ost=sys.stderr, unit='G', mode='md')

cch.show_summary_report(ost=sys.stderr, unit='G', mode='md')
```


#### Prefixed-units

The following unit prefixes are supported by `unit` argument of
`show_report` and `show_summary_report`.

* `None`: if you want to use raw values
* `K`: 10^3 (for FLOPs) or 1024^1 (for memory report)
* `M`: 10^6 or 1024^2
* `G`: 10^9 or 1024^3
* `T`: 10^12 or 1024^4

For memory report, the unit will be shown as like `KiB` or `MiB` instead of
`KB`.


#### Number of digits

You can specify how many digits after the decimal point to show to
`n_digits` argument of `show_report` and `show_summary_report`.

By default it is set to 3.
Possible value is between 0 (round to integer) to 10.
If `None` is specified it is treated as 10.

Be noted that you do not need to worry about numerical error in summary report
due to the rounding, because summary values are calculated before rounding.

```
>>> cch.show_summary_report(unit='G', mode='table', n_digits=8)
+-----------------------+----------+-------------+------------+------------+------------+
|      Layer type       | # Layers |   GFLOPs    |  MemRead   |  MemWrite  |   MemR+W   |
|                       |          |             |    GiB     |    GiB     |    GiB     |
+=======================+==========+=============+============+============+============+
| Convolution2DFunction | 13       | 15.34663066 | 0.08864903 | 0.05046844 | 0.13911748 |
+-----------------------+----------+-------------+------------+------------+------------+
| ReLU                  | 15       | 0.01355571  | 0.05049896 | 0.05049896 | 0.10099792 |
+-----------------------+----------+-------------+------------+------------+------------+
| ...                   | ...      | ...         | ...        | ...        | ...        |
```


#### Custom columns

You can specify which column to show as a table to
`columns` argument of `show_report` and `show_summary_report`.

There are two ways to customize columns.

The first way is to make use of predefined columns set.
There are some column definitions in `SummaryColumns` for
`show_summary_report`, and `ReportColumns` for `show_report`, respectively.

```
>>> cch.show_summary_report(unit='G', mode='table', columns=SummaryColumns.ALL)
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
|      Layer type       | # Layers | GFLOPs | MemRead | MemWrite | MemR+W |  FLOPs  | MemRead | MemWrite | MemR+W  |
|                       |          |        |   GiB   |   GiB    |  GiB   |   (%)   |   (%)   |   (%)    |   (%)   |
+=======================+==========+========+=========+==========+========+=========+=========+==========+=========+
| Convolution2DFunction | 13       | 15.347 | 0.089   | 0.05     | 0.139  | 99.085% | 14.237% | 47.297%  | 19.073% |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
| ReLU                  | 15       | 0.014  | 0.05    | 0.05     | 0.101  | 0.088%  | 8.11%   | 47.325%  | 13.847% |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
| MaxPooling2D          | 5        | 0.005  | 0.023   | 0.006    | 0.029  | 0.03%   | 3.662%  | 5.343%   | 3.908%  |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
| Reshape               | 1        | 0.0    | 0.0     | 0.0      | 0.0    | 0.0%    | 0.0%    | 0.0%     | 0.0%    |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
| LinearFunction        | 3        | 0.124  | 0.461   | 0.0      | 0.461  | 0.798%  | 73.991% | 0.032%   | 63.171% |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
| Softmax               | 1        | 0.0    | 0.0     | 0.0      | 0.0    | 0.0%    | 0.001%  | 0.003%   | 0.001%  |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
| total                 | 38       | 15.488 | 0.623   | 0.107    | 0.729  | 100.0%  | 100.0%  | 100.0%   | 100.0%  |
+-----------------------+----------+--------+---------+----------+--------+---------+---------+----------+---------+
```

The other way is to manually specify the column list.

```
>>> cch.show_report(unit='G', mode='table' , columns=[
...     'name', 'flops', 'mread', 'mwrite', 'mrw', 'output_shapes', "params"
... ])
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
|        Layer name        | GFLOPs | MemRead | MemWrite | MemR+W |    Output shapes     |            Function parameters             |
|                          |        |   GiB   |   GiB    |  GiB   |                      |                                            |
+==========================+========+=========+==========+========+======================+============================================+
| Convolution2DFunction-1  | 0.087  | 0.001   | 0.012    | 0.013  | [(1, 64, 224, 224)]  | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-1                   | 0.003  | 0.012   | 0.012    | 0.024  | [(1, 64, 224, 224)]  |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-2  | 1.850  | 0.012   | 0.012    | 0.024  | [(1, 64, 224, 224)]  | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ...                      | ...    | ...     | ...      | ...    | ...                  | ...                                        |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| total                    | 15.488 | 0.623   | 0.107    | 0.729  |                      |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
```


### Access to the detailed report

Once you let `cch` gather the computational costs as explained above,
you can access to the report information directly.

```python
>>> cch.layer_report
```

It is a huge dictionary whose structure is:

```
{
    "Convolution2DFunction-1": {
        "name": "Convolution2DFunction-1",
        "type": "Convolution2DFunction",
        "flops": 86704128,
        "mread": 609280,
        "mwrite": 12845056,
        "mrw": 13454336,
        "traceback": "...",
        "input_shapes": [[1, 3, 224, 224], [64, 3, 3, 3], [ 4]],
        "output_shapes": [[1, 64, 224, 224]],
        "params": {"k": 3, "s": 1, "p": 1, "d": 1, "groups": 1, "nobias": false},
        "flops%": 0.5597995752663483,
        "mread%": 0.09112725854650683,
        "mwrite%": 11.21102960110868,
        "mrw%": 1.7179140987382209
    },
    "ReLU-1": {
        "name": "ReLU-1",
        "type": "ReLU",
        "flops": 3211264,
        "mread": 12845056,
        "mwrite": 12845056,
        "mrw": 25690112,
        "traceback": ...,
        "input_shapes": [[1, 64, 224, 224]],
        "output_shapes": [[1, 64, 224, 224]],
        "params": {},
        "flops%": 0.020733317602457342,
        "mread%": 1.9211770272392967,
        "mwrite%": 11.21102960110868,
        "mrw%": 3.2802366168768162
    },
    ...
}
```


Also, summary report can be found.
This contains total costs for each type of layers.

```python
>>> cch.summary_report
{
    "Convolution2DFunction": {
        "type": "Convolution2DFunction",
        "name": "Convolution2DFunction",
        "flops": 15346630656,
        "n_layers": 13,
        "mread": 95186176,
        "mwrite": 54190080,
        "mrw": 149376256,
        "flops%": 99.08452482214363,
        "mread%": 14.236566554630553,
        "mwrite%": 47.29653112967724,
        "mrw%": 19.07307623350047
    },
    "ReLU": {
        "type": "ReLU",
        "name": "ReLU",
        "flops": 13555712,
        "n_layers": 15,
        "mread": 54222848,
        "mwrite": 54222848,
        "mrw": 108445696,
        "flops%": 0.08752157475169971,
        "mread%": 8.109866545469965,
        "mwrite%": 47.32513069498619,
        "mrw%": 13.846866178002324
    },
    ...
    "total": {
        "name": "total",
        "type": "total",
        "flops": 15488423327,
        "n_layers": 38,
        "mread": 668603456,
        "mwrite": 114575168,
        "mrw": 783178624,
        "flops%": 100.0,
        "mread%": 100.0,
        "mwrite%": 100.0,
        "mrw%": 100.0
    }
}
```


### Supported layers

Please see [DETAILS.md](DETAILS.md).


### Custom cost calculator for non-supported layer types

Layer types supported are listed in the next section.

In case you need an unsupported layer or you have your custom layer,
you can insert a cost calculator.

```python
def custom_calculator(func, in_data, **kwargs)
    ...
    return (0, 0, 0)

with chainer.no_backprop_mode(), chainer.using_config('train', False):
    with ComputationalCostHook(fma_1flop=True) as cch:
        cch.add_custom_cost_calculator(F.math.basic_math.Add, custom_calculator)
        y = x + x   # Call Add

        cch.report['Add-0']    # you can find custom estimation result
```

A custom cost calculator has to have the following signature.
* First positional argument:
  * A `Function` or `FunctionNode` object will be passed.
* Second positional argument
  * List of data (could be `numpy.array`, `cupy.array` or a scalar) will be fed
* Third keyword argument
  * Some flags will be fed
    * `fma_1flop: bool`, for example

Also, a calculator has to return a tuple with the following 4 elements:
* Number of FLOPs in `int`
  * Focus only on principle floating point operations
  * If `fma_1flop=True` is specified in `kwargs`,
    please treat operations that can be fused to an FMA operation as 1 FLOP.
* Memory read (number of elements) in `int`
* Memory write (number of elements) in `int`
  * Not in "bytes". Conversion to bytes is automatically done by
    `ComputationalCostHook`.
* Parameters to report in `dict`
  * Function-specific parameters, e.g. `k` of `Convolution2DFunction`.

For more details about how to implement custom cost calculator,
please refer existing implementations located in
`chainer_computational_cost/cost_calculators/*.py`.

You can overwrite your custom calculator to existing one.
This is useful when for example the device or environment you're considering
has some special specifications that are different from normal behavior.
e.g. there is an inference engine that doesn't support inplace `Reshape`,
whose `mread` and `mwrite` won't be 0.


If a layer not supported by chainer_computational_cost is used,
it shows a message like
`Warning: XXXFunction is not yet supported by ComputationalCostHook, ignored`.

Also, you can access to which layers are ignored.

```python
with ComputationalCostHook() as cch:
  ...
  print(cch.ignored_layers)
```

It has the following structure.

```
{
  'XXXFunction':
  {
    'type': 'XXXFunction',
    'traceback': '(traceback)'
  },
  ...
}
```


## Contribution Guide

### New layer support

Adding layer type support is one of the most important contribution.

The specification is almost same as cost calculator for custom layers
explaine above.

In addition, functions have to follow the following rules.

* Name should be `calc_xxxx`
* Source code name has to correspond to chainer's file arrangement
  * For example, BN is in [chainer/functions/normalization](https://github.com/chainer/chainer/tree/v4.0.0/chainer/functions/normalization),
    so the cost calculator is implemented in `chainer_computational_cost/cost_calculators/normalization.py`.

Once you properly implemented your cost calculation function,
chainer-computational-cost automatically find and configure.
This auto-discovery mechanism is implemented in
`chainer_computational_cost/cost_calculators/__init__.py`.

Also, please write a docstring for each cost calculator in Markdown format.
It will be used to generate DETAILS.md.
Please refer Documentation section below for more details.


### Coding standard

Following the [Chainer contribution guide](https://docs.chainer.org/en/stable/contribution.html),
chainer-computational-cost also requires flake8 and hacking.

```python
% pip install hacking flake8
% flake8 .
```


### Testing

We use pytest for unit-testing.
Please write sufficient test cases when you support new functions
or implement a new feataure.

```bash
% pip install pytest
% python -m pytest
```

In order to check coverage locally, please confirm by following command.

```bash
% pip install pytest-cov
% python -m pytest --cov chainer_computational_cost --cov-report html:cov_html
% open cov_html/index.html
```

Every PRs will be automatically tested by
[Travis CI](https://travis-ci.org/belltailjp/chainer_computational_cost)
and code coverage of the test is monitored by
[coveralls](https://coveralls.io/github/belltailjp/chainer_computational_cost).
Please make sure your PR becomes all green.


### Documentation

(TODO: consider better way)

DETAILS.md is automatically generated by make_details_md.py script.

This script collects docstring of each cost calculation functions.
If a cost calculator doesn't have a docstring, it won't appear in DETAILS.md.
So please always write docstring for every cost calculators
in the following format.

```python
def calc_xxxx(func: XXXX, inputs, **kwargs):
    """XXXX

    XXXX is defined as: $y=f(x)$
    ...
    |Item|Value|
    |:---|:---|
    |FLOPs| $$ 4 \| x \| $$ |
    |mread| $$ \| x \| $$ |
    |mwrite| $$ \| x \| $$ |
    """
```

* It supports markdown format
* First line should be function name (*No newline* between `"""` and `XXX`)
* 1-line blank
* Detailed explanation
* It is suggested to summarize as a table.

Also, mathematical formula is supported. This is an example of inline equation.

```
function $y=f(x)$ is ...
```

And non-inline equation.

```
function is defined a follows:
$$y=f(x)$$
```

Both inline and non-inline formula, please write in one line.

```
# NG pattern
$$
y=f(x)
$$
```

After writing docstring, DETAILS.md can be generated by:

```bash
% python make_details_md.py
```

Then please don't forget to commit new DETAILS.md.


## Discriminator

There is not assurance of cost estimation formula, and it might change in the
future.
Please verify it by yourself if you will use this for critical purposes.


## Acknowledgements

The key concept of chainer-computational-cost is originally developed by
[t-abe](https://github.com/t-abe).
