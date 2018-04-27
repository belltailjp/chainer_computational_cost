# chainer-computational-cost

This is a tool to estimate theoretical computational cost of a chainer-based neural network.

## Requirements

* chainer >= 4.0.0


## Installation

```bash
% python setup.py install
```

## Quick Start

```python
import numpy as np
import chainer
import chainer.links as L
from chainer_computational_cost import ComputationalCostHook

net = L.VGG16Layers()
x = np.random.random((1, 3, 224, 224)).astype(np.float32)
with chainer.no_backprop_mode(), chainer.using_config('train', False):
    with ComputationalCostHook(unify_fma=True) as cost:
        y = net(x)
        cost.show_report(unit='G', mode='md')
```


It will show the following table to stdout.

|layer|GOPS|mread(GB)|mwrite(GB)|
|:----|:----|:----|:----|
|Convolution2DFunction-0|0.089915392|0.00060928|
|ReLU-0|0.003211264|0.012845056|
|Convolution2DFunction-1|1.852899328|0.012992768|
|ReLU-1|0.003211264|0.012845056|
|MaxPooling2D-0|0.002408448|0.012845056|
|Convolution2DFunction-2|0.926449664|0.003506688|
|ReLU-2|0.001605632|0.006422528|
|Convolution2DFunction-3|1.851293696|0.007012864|
|ReLU-3|0.001605632|0.006422528|
|MaxPooling2D-1|0.001204224|0.006422528|
|Convolution2DFunction-4|0.925646848|0.002786304|
|ReLU-4|0.000802816|0.003211264|
|Convolution2DFunction-5|1.85049088|0.005571584|
|ReLU-5|0.000802816|0.003211264|
|Convolution2DFunction-6|1.85049088|0.005571584|
|ReLU-6|0.000802816|0.003211264|
|MaxPooling2D-2|0.000602112|0.003211264|
|Convolution2DFunction-7|0.92524544|0.005523456|
|ReLU-7|0.000401408|0.001605632|
|Convolution2DFunction-8|1.850089472|0.011044864|
|ReLU-8|0.000401408|0.001605632|
|Convolution2DFunction-9|1.850089472|0.011044864|
|ReLU-9|0.000401408|0.001605632|
|MaxPooling2D-3|0.000301056|0.001605632|
|Convolution2DFunction-10|0.462522368|0.00984064|
|ReLU-10|0.000100352|0.000401408|
|Convolution2DFunction-11|0.462522368|0.00984064|
|ReLU-11|0.000100352|0.000401408|
|Convolution2DFunction-12|0.462522368|0.00984064|
|ReLU-12|0.000100352|0.000401408|
|MaxPooling2D-4|7.5264e-05|0.000401408|
|Reshape-0|0.0|0.0|
|LinearFunction-0|0.102764544|0.411158528|
|ReLU-13|4.096e-06|1.6384e-05|
|LinearFunction-1|0.016781312|0.067141632|
|ReLU-14|4.096e-06|1.6384e-05|
|LinearFunction-2|0.004097|0.016404384|
|total|15.501967848|0.668599456|



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

The prefix will affect to both OPS, and memory report.


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


