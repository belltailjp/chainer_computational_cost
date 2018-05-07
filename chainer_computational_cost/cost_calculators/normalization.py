from chainer.functions.normalization.batch_normalization \
        import FixedBatchNormalization


def calc_fixed_bn(func: FixedBatchNormalization, in_data, **kwargs):
    x, _, _, mean, var = in_data
    x = in_data[0]
    n_elements = len(x.flatten())
    ops = n_elements * 2    # *2 <- scale and shift
    mread = n_elements + len(mean) + len(var)
    mwrite = n_elements
    return (ops, mread, mwrite)
