# chainer-computational-cost

This is a tool to estimate theoretical computational cost
of a chainer-based neural network.

You can analyze

* Theoretical amount of floating point arithmetics (FLOPs)
* Theoretical amount of memory read and write (mread/mwrite)

For each layer.

Also, summary of these computational costs for each layer-type,
and total cost can be calculated.


## Requirements

* python >= 3
* chainer >= 4.0.0


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
    with ComputationalCostHook(unify_fma=True) as cost:
        y = net(x)
        cost.show_report(unit='G', mode='md')
```

It will show the following table to stdout.

|Layer name|GFLOPs|MemRead GB/s|MemWrite GB/s|MemR/W GB/s|
|:----|:----|:----|:----|:----|
|Convolution2DFunction-1|0.089915392|0.00060928|0.012845056|0.013454336|
|ReLU-1|0.003211264|0.012845056|0.012845056|0.025690112|
|Convolution2DFunction-2|1.852899328|0.012992768|0.012845056|0.025837824|
|ReLU-2|0.003211264|0.012845056|0.012845056|0.025690112|
|MaxPooling2D-1|0.002408448|0.012845056|0.003211264|0.01605632|
|Convolution2DFunction-3|0.926449664|0.003506688|0.006422528|0.009929216|
|ReLU-3|0.001605632|0.006422528|0.006422528|0.012845056|
|Convolution2DFunction-4|1.851293696|0.007012864|0.006422528|0.013435392|
|ReLU-4|0.001605632|0.006422528|0.006422528|0.012845056|
|MaxPooling2D-2|0.001204224|0.006422528|0.001605632|0.00802816|
|Convolution2DFunction-5|0.925646848|0.002786304|0.003211264|0.005997568|
|ReLU-5|0.000802816|0.003211264|0.003211264|0.006422528|
|Convolution2DFunction-6|1.85049088|0.005571584|0.003211264|0.008782848|
|ReLU-6|0.000802816|0.003211264|0.003211264|0.006422528|
|Convolution2DFunction-7|1.85049088|0.005571584|0.003211264|0.008782848|
|ReLU-7|0.000802816|0.003211264|0.003211264|0.006422528|
|MaxPooling2D-3|0.000602112|0.003211264|0.000802816|0.00401408|
|Convolution2DFunction-8|0.92524544|0.005523456|0.001605632|0.007129088|
|ReLU-8|0.000401408|0.001605632|0.001605632|0.003211264|
|Convolution2DFunction-9|1.850089472|0.011044864|0.001605632|0.012650496|
|ReLU-9|0.000401408|0.001605632|0.001605632|0.003211264|
|Convolution2DFunction-10|1.850089472|0.011044864|0.001605632|0.012650496|
|ReLU-10|0.000401408|0.001605632|0.001605632|0.003211264|
|MaxPooling2D-4|0.000301056|0.001605632|0.000401408|0.00200704|
|Convolution2DFunction-11|0.462522368|0.00984064|0.000401408|0.010242048|
|ReLU-11|0.000100352|0.000401408|0.000401408|0.000802816|
|Convolution2DFunction-12|0.462522368|0.00984064|0.000401408|0.010242048|
|ReLU-12|0.000100352|0.000401408|0.000401408|0.000802816|
|Convolution2DFunction-13|0.462522368|0.00984064|0.000401408|0.010242048|
|ReLU-13|0.000100352|0.000401408|0.000401408|0.000802816|
|MaxPooling2D-5|7.5264e-05|0.000401408|0.000100352|0.00050176|
|Reshape-1|0.0|0.0|0.0|0.0|
|LinearFunction-1|0.102764544|0.411158528|1.6384e-05|0.411174912|
|ReLU-14|4.096e-06|1.6384e-05|1.6384e-05|3.2768e-05|
|LinearFunction-2|0.016781312|0.067141632|1.6384e-05|0.067158016|
|ReLU-15|4.096e-06|1.6384e-05|1.6384e-05|3.2768e-05|
|LinearFunction-3|0.004097|0.016404384|4e-06|0.016408384|
|Softmax-1|2.999e-06|4e-06|4e-06|8e-06|
|total|15.501970847|0.668603456|0.114575168|0.783178624|



If you call `show_summary_report` method,
it will show summary for each type of layer.

Both `show_report` and `show_summary_report` support
`mode` param to switch print mode.
Default is `mode='csv'`, but `mode='md'` (markdown table mode) and
`mode='table'` (text table) are also supported.

```
>>> cost.show_summary_report(unit='G', mode='table')
+-----------------------+-----------+----------+----------+----------+
|      Layer type       |  GFLOPs   | MemRead  | MemWrite |  MemR/W  |
|                       |           |   GB/s   |   GB/s   |   GB/s   |
+=======================+===========+==========+==========+==========+
| Convolution2DFunction | 15.360178 | 0.095186 | 0.054190 | 0.149376 |
+-----------------------+-----------+----------+----------+----------+
| ReLU                  | 0.013556  | 0.054223 | 0.054223 | 0.108446 |
+-----------------------+-----------+----------+----------+----------+
| MaxPooling2D          | 0.004591  | 0.024486 | 0.006121 | 0.030607 |
+-----------------------+-----------+----------+----------+----------+
| Reshape               | 0         | 0        | 0        | 0        |
+-----------------------+-----------+----------+----------+----------+
| LinearFunction        | 0.123643  | 0.494705 | 0.000037 | 0.494741 |
+-----------------------+-----------+----------+----------+----------+
| Softmax               | 0.000003  | 0.000004 | 0.000004 | 0.000008 |
+-----------------------+-----------+----------+----------+----------+
| total                 | 15.501971 | 0.668603 | 0.114575 | 0.783179 |
+-----------------------+-----------+----------+----------+----------+
```


In addition, you can specify which column to show as a table.

```
>>> cost.show_report(unit='G', mode='table' , columns=[
...     'name', 'flops', 'mread', 'mwrite', 'mrw', 'output_shapes', "params"
... ])
+--------------------------+-----------+----------+----------+----------+----------------------+--------------------------------------------+
|        Layer name        |  GFLOPs   | MemRead  | MemWrite |  MemR/W  |    Output shapes     |            Function parameters             |
|                          |           |   GB/s   |   GB/s   |   GB/s   |                      |                                            |
+==========================+===========+==========+==========+==========+======================+============================================+
| Convolution2DFunction-1  | 0.089915  | 0.000609 | 0.012845 | 0.013454 | [(1, 64, 224, 224)]  | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+-----------+----------+----------+----------+----------------------+--------------------------------------------+
| ReLU-1                   | 0.003211  | 0.012845 | 0.012845 | 0.025690 | [(1, 64, 224, 224)]  |                                            |
+--------------------------+-----------+----------+----------+----------+----------------------+--------------------------------------------+
| Convolution2DFunction-2  | 1.852899  | 0.012993 | 0.012845 | 0.025838 | [(1, 64, 224, 224)]  | k=3, s=1, p=1, d=1, groups=1, nobias=False |
+--------------------------+-----------+----------+----------+----------+----------------------+--------------------------------------------+
| ReLU-2                   | 0.003211  | 0.012845 | 0.012845 | 0.025690 | [(1, 64, 224, 224)]  |                                            |
+--------------------------+-----------+----------+----------+----------+----------------------+--------------------------------------------+
| MaxPooling2D-1           | 0.002408  | 0.012845 | 0.003211 | 0.016056 | [(1, 64, 112, 112)]  | k=2, s=2, p=0                              |
+--------------------------+-----------+----------+----------+----------+----------------------+--------------------------------------------+
| ...                      | ...       | ...      | ...      | ...      | ...                  | ...                                        |

```


## Usage

As for basic usage, please refer to the avobe quickstart.


### Unify FMA mode

When `unify_fma` is set to `True`, chainer_computational_cost considers
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
* `k`: 10^3
* `M`: 10^6
* `G`: 10^9
* `T`: 10^12

The prefix will affect to both FLOPs, and memory report.


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
        "flops": 89915392,
        "mread": 609280,
        "mwrite": 12845056,
        "mrw": 13454336,
        "traceback": "(traceback string)",
        "input_shapes": [[1, 3, 224, 224], [64, 3, 3, 3], [64]],
        "output_shapes": [[1, 64, 224, 224]],
        "params": {"k": 3, "s": 1, "p": 1, "d": 1, "groups": 1, "nobias": false}
    },
    "ReLU-1": {
        "name": "ReLU-1",
        "type": "ReLU",
        "flops": 3211264,
        "mread": 12845056,
        "mwrite": 12845056,
        "mrw": 25690112,
        "traceback": "(traceback)",
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
        "flops": 15501970847,
        "mread": 668603456,
        "mwrite": 114575168,
        "mrw": 783178624,
        "input_shapes": "--",
        "output_shapes": "--",
        "params": {}
    },
    "Convolution2DFunction": {
        "type": "Convolution2DFunction",
        "name": "total",
        "flops": 15360178176,
        "mread": 95186176,
        "mwrite": 54190080,
        "mrw": 149376256,
        "input_shapes": "--",
        "output_shapes": "--",
        "params": {}
    },
    "ReLU": {
        "type": "ReLU",
        "name": "total",
        "flops": 13555712,
        "mread": 54222848,
        "mwrite": 54222848,
        "mrw": 108445696,
        "input_shapes": "--",
        "output_shapes": "--",
        "params": {}
    },
	  ...
}
```


### Custom cost calculator for non-supported layer types

Layer types supported are listed in the next section.

In case you need an unsupported layer or you have your custom layer,
you can insert a cost calculator.

```python
def custom_calculator(func: F.math.basic_math.Add, in_data, **kwargs)
    ...
    return (0, 0, 0)

with chainer.no_backprop_mode(), chainer.using_config('train', False):
    with ComputationalCostHook(unify_fma=True) as cost:
        cost.add_custom_cost_calculator(custom_calculator)
        y = x + x   # Call Add

        cost.report['Add-0']    # you can find custom estimation result
```

Custom cost calculator has to have the following signature.
* First positional argument
  * Name: `func`
  * Type annotation is required.
    Specify proper type which you want to calculate by the function.
    Type should be a subclass of `FunctionNode`.
* Second positional argument
  * Name: `in_data`
  * List of data (could be `numpy.array`, `cupy.array` or a scalar) will be fed
* Third keyword argument
  * Name: `**kwargs`
  * Some flags will be fed
    * `unify_fma: bool`, for example

Also, a calculator has to return a tuple with the following 4 elements:
* Number of FLOPs in `int`
  * Focus only on principle floating point operations
  * If `unify_fma=True` is specified in `kwargs`,
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
This is useful when the device or environment you're considering has
some particular specifications that are different.
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


### Supported functions

* Activation
  * ReLU
  * Sigmoid
* Array
  * Reshape
  * ResizeImages
  * Shift
* Connection
  * Concat
  * Convolution2D
  * Deconvolution2D
  * Linear
* Math
  * Add, Div, Mul, Sub
* Pooling
  * AveragePooling2D
  * MaxPooling2D
* Normalization
  * FixedBatchNormalization


## Contribution Guide

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
