def calc_reshape(function, in_data, **kwargs):
    size = in_data[0].size
    return (0, 0, 0)


def calc_concat(function, in_data, **kwargs):
    size = sum([x.size for x in in_data])
    return (0, size, size)


def calc_resize(function, in_data, **kwargs):
    x, = in_data
    batch_size, in_c = x.shape[:2]
    out_size = batch_size * in_c * function.out_H * function.out_W
    return (out_size * 18, out_size * 4, out_size)


def calc_transpose(function, in_data, **kwargs):
    x, = in_data
    return (0, x.size, x.size)
