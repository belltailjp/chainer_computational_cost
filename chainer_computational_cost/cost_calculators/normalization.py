from chainer.functions.normalization.batch_normalization \
        import FixedBatchNormalization
from chainer.functions.normalization.local_response_normalization \
        import LocalResponseNormalization


def calc_fixed_bn(func: FixedBatchNormalization, in_data, **kwargs):
    x, _, _, mean, var = in_data
    x = in_data[0]
    n_elements = len(x.flatten())
    ops = n_elements * 2    # *2 <- scale and shift
    mread = n_elements + len(mean) + len(var)
    mwrite = n_elements
    return (ops, mread, mwrite)


def calc_lrn(func: LocalResponseNormalization, in_data, **kwargs):
    x, = in_data

    c = x.shape[1]
    s = c * func.k - (func.k // 2) * 2
    ops_square = x.size
    ops_neighbor_sum = x.size * s + x.size * 3
    ops_total = ops_square + ops_neighbor_sum + x.size
    ops = ops_total
    mread = x.size + x.shape[1] * x.shape[2] * s
    mwrite = x.size
    return (ops, mread, mwrite)
