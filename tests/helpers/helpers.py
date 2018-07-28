from chainer_computational_cost.cost_calculators import calculators


def calculate_cost(f, inputs, **kwargs):
    """Find registered cost calculator and calls it by a given array of inputs

    Args:
        f: Chainer's Function or FunctionNode object
        inputs: A list input values
        kwargs: Keyword arguments that the cost calculator can receive, for
          example, fma_1flop
    """
    flops, mread, mwrite, params = calculators[type(f)](f, inputs, **kwargs)
    assert type(flops) is int
    assert type(mread) is int
    assert type(mwrite) is int
    assert type(params) is dict
    return (flops, mread, mwrite, params)
