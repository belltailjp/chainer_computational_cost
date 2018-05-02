from . import *

calculators = {
    '_ + _': calc_eltw_op,
    '_ - _': calc_eltw_op,
    '_ * _': calc_eltw_op,
    '_ / _': calc_eltw_op,
    'LinearFunction': calc_linear,
    'Convolution2DFunction': calc_conv2d,
    'ReLU': calc_activation,
    'Sigmoid': calc_activation,
    'LeakyReLU': calc_activation,
    'PReLU': calc_prelu,
    'FixedBatchNormalization': calc_fixed_bn,
    'Reshape': calc_reshape,
    'MaxPooling2D': calc_max_pooling2d,
    'AveragePooling2D': calc_average_pooling2d,
    'ResizeImages': calc_resize,
    'Concat': calc_concat,
    'Shift': calc_shift
}
