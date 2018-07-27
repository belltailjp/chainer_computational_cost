# chainer-computational-cost

This is a tool to estimate theoretical computational cost
of a chainer-based neural network.

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

* python 3.6 or 2.7
* chainer >= 4.0.0
* (optional) texttable >= 1.4.0
* (optional) pytest >= 3.5.1


## Installation

```bash
% python setup.py install
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
    with ComputationalCostHook(fma_1flop=True) as cost:
        y = net(x)
        cost.show_report(unit='G', mode='md')
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

Both `show_report` and `show_summary_report` support
`mode` param to switch print mode.
Default is `mode='csv'`, but `mode='md'` (markdown table mode) and
`mode='table'` (text table) are also supported.

```
>>> cost.show_summary_report(unit='G', mode='table')
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


In addition, you can specify which column to show as a table.

```
>>> cost.show_report(unit='G', mode='table' , columns=[
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
| ReLU-2                   | 0.003  | 0.012   | 0.012    | 0.024  | [(1, 64, 224, 224)]  |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| MaxPooling2D-1           | 0.002  | 0.012   | 0.003    | 0.015  | [(1, 64, 112, 112)]  | k=2, s=2, p=0                              |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-3  | 0.925  | 0.003   | 0.006    | 0.009  | [(1, 128, 112, 112)] | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-3                   | 0.002  | 0.006   | 0.006    | 0.012  | [(1, 128, 112, 112)] |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-4  | 1.850  | 0.007   | 0.006    | 0.013  | [(1, 128, 112, 112)] | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-4                   | 0.002  | 0.006   | 0.006    | 0.012  | [(1, 128, 112, 112)] |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| MaxPooling2D-2           | 0.001  | 0.006   | 0.001    | 0.007  | [(1, 128, 56, 56)]   | k=2, s=2, p=0                              |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-5  | 0.925  | 0.003   | 0.003    | 0.006  | [(1, 256, 56, 56)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-5                   | 0.001  | 0.003   | 0.003    | 0.006  | [(1, 256, 56, 56)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-6  | 1.850  | 0.005   | 0.003    | 0.008  | [(1, 256, 56, 56)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-6                   | 0.001  | 0.003   | 0.003    | 0.006  | [(1, 256, 56, 56)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-7  | 1.850  | 0.005   | 0.003    | 0.008  | [(1, 256, 56, 56)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-7                   | 0.001  | 0.003   | 0.003    | 0.006  | [(1, 256, 56, 56)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| MaxPooling2D-3           | 0.001  | 0.003   | 0.001    | 0.004  | [(1, 256, 28, 28)]   | k=2, s=2, p=0                              |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-8  | 0.925  | 0.005   | 0.001    | 0.007  | [(1, 512, 28, 28)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-8                   | 0      | 0.001   | 0.001    | 0.003  | [(1, 512, 28, 28)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-9  | 1.850  | 0.010   | 0.001    | 0.012  | [(1, 512, 28, 28)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-9                   | 0      | 0.001   | 0.001    | 0.003  | [(1, 512, 28, 28)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-10 | 1.850  | 0.010   | 0.001    | 0.012  | [(1, 512, 28, 28)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-10                  | 0      | 0.001   | 0.001    | 0.003  | [(1, 512, 28, 28)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| MaxPooling2D-4           | 0      | 0.001   | 0        | 0.002  | [(1, 512, 14, 14)]   | k=2, s=2, p=0                              |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-11 | 0.462  | 0.009   | 0        | 0.010  | [(1, 512, 14, 14)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-11                  | 0      | 0       | 0        | 0.001  | [(1, 512, 14, 14)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-12 | 0.462  | 0.009   | 0        | 0.010  | [(1, 512, 14, 14)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-12                  | 0      | 0       | 0        | 0.001  | [(1, 512, 14, 14)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Convolution2DFunction-13 | 0.462  | 0.009   | 0        | 0.010  | [(1, 512, 14, 14)]   | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-13                  | 0      | 0       | 0        | 0.001  | [(1, 512, 14, 14)]   |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| MaxPooling2D-5           | 0      | 0       | 0        | 0      | [(1, 512, 7, 7)]     | k=2, s=2, p=0                              |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Reshape-1                | 0      | 0       | 0        | 0      | [(1, 25088)]         | in_shape=(1, 512, 7, 7), out_shape=(1, -1) |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| LinearFunction-1         | 0.103  | 0.383   | 0        | 0.383  | [(1, 4096)]          | nobias=False                               |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-14                  | 0      | 0       | 0        | 0      | [(1, 4096)]          |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| LinearFunction-2         | 0.017  | 0.063   | 0        | 0.063  | [(1, 4096)]          | nobias=False                               |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| ReLU-15                  | 0      | 0       | 0        | 0      | [(1, 4096)]          |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| LinearFunction-3         | 0.004  | 0.015   | 0        | 0.015  | [(1, 1000)]          | nobias=False                               |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| Softmax-1                | 0      | 0       | 0        | 0      | [(1, 1000)]          | axis=1                                     |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
| total                    | 15.488 | 0.623   | 0.107    | 0.729  |                      |                                            |
+--------------------------+--------+---------+----------+--------+----------------------+--------------------------------------------+
```


## Usage

As for basic usage, please refer to the avobe quickstart.


### Unify FMA mode

When `fma_1flop` is set to `True`, chainer_computational_cost considers
FMA (fused multiply and add, `ax + b`) as one operation.
Otherwise, it counts as 2 operations.

This affects to convolution and linear layers' estimation.


### Reporting

Estimated computational cost table is reported by calling `show_report` method.

Currently it supports the following modes.

* CSV mode (`mode='csv'`)
* Markdown table (`mode='md'`)
* Prettified table (`mode='table'`)

By default, report is written to stdout.
You can specify stream (e.g. file object) to `dst` argument.

```python
cost.show_report(ost=sys.stderr, unit='G', mode='md')
```

Also, the following unit prefixes are supported by `unit` argument.

* `None`: if you want to use raw values
* `K`: 10^3 (for FLOPs) or 1024^1 (for memory report)
* `M`: 10^6 or 1024^2
* `G`: 10^9 or 1024^3
* `T`: 10^12 or 1024^4

For memory report, the unit will be shown as like `Ki` or `Mi`.

In addition, you can specify how many digits after the decimal point to show to
`n_digits` argument.
By default it's 3, possible values are between 0 (round to integer) to 10, and
if `None` is specified it is treated as 10.
Be noted that you do not need to worry about numerical error in summary report
due to the rounding, because summary values are calculated before rounding.


### Access to the detailed report

Once you let `cost` gather the computational costs as explained above,
you can access to the report information directly.

```python
>>> cost.layer_report
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
        "traceback": ...,
        "input_shapes": [[1, 3, 224, 224], [64, 3, 3, 3],[64]],
        "output_shapes": [[1, 64, 224, 224]],
        "params": {
            "k": 3,
            "s": 1,
            "p": 1,
            "d": 1,
            "groups": 1,
            "nobias": false
        }
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
        "params": {}
    },
	  ...
}
```


Also, summary report can be found.
This contains total costs for each type of layers.

```python
>>> cost.summary_report
{
    "total": {
        "type": "total",
        "name": "total",
        "n_layers": 38,
        "flops": 15488423327,
        "mread": 668603456,
        "mwrite": 114575168,
        "mrw": 783178624
    },
    "Convolution2DFunction": {
        "type": "Convolution2DFunction",
        "name": "Convolution2DFunction",
        "n_layers": 13,
        "flops": 15346630656,
        "mread": 95186176,
        "mwrite": 54190080,
        "mrw": 149376256
    },
    "ReLU": {
        "type": "ReLU",
        "name": "ReLU",
        "n_layers": 15,
        "flops": 13555712,
        "mread": 54222848,
        "mwrite": 54222848,
        "mrw": 108445696
    },
	  ...
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
    with ComputationalCostHook(fma_1flop=True) as cost:
        cost.add_custom_cost_calculator(F.math.basic_math.Add, custom_calculator)
        y = x + x   # Call Add

        cost.report['Add-0']    # you can find custom estimation result
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
with ComputationalCostHook() as cost:
  ...
  print(cost.ignored_layers)
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

```python
% pip install pytest
% python -m pytest
```


### Documentation

DETAILS.md is automatically generated by make_details_md.py script.

This script collects docstring of each cost calculation functions.
If a cost calculator doesn't have docstring, it won't appear in DETAILS.md.
So please always write docstring for every cost calculators
in the following format.

```python
def calc_xxxx(func: XXXX, inputs, **kwargs):
    """XXXX

    XXXX is defined as: $y=f(x)$
    ...
    |Item|Value|
    |:---|:---|
    |FLOPs|4 * size of input|
    |mread|size of input|
    |mwrite|size of input|
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
