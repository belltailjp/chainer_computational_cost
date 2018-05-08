from chainer.functions.normalization.batch_normalization \
    import FixedBatchNormalization
from chainer.functions.normalization.local_response_normalization \
    import LocalResponseNormalization


def calc_fixed_bn(func: FixedBatchNormalization, in_data, **kwargs):
    x, _, _, mean, var = in_data
    x = in_data[0]
    n_elements = len(x.flatten())
    flops = n_elements * 2    # *2 <- scale and shift
    mread = n_elements + len(mean) + len(var)
    mwrite = n_elements
    return (flops, mread, mwrite)


def calc_lrn(func: LocalResponseNormalization, in_data, **kwargs):
    x, = in_data

    c = x.shape[1]
    s = c * func.k - (func.k // 2) * 2
    flops_square = x.size
    flops_neighbor_sum = x.size * s + x.size * 3
    flops_total = flops_square + flops_neighbor_sum + x.size
    flops = flops_total
    mread = x.size + x.shape[1] * x.shape[2] * s
    mwrite = x.size
    return (flops, mread, mwrite)
