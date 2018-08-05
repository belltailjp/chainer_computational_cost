from distutils.version import LooseVersion
import importlib
import warnings

import chainer


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


def require_import(func_fqn):
    """Decorator to turn on/off a test case by existence of a class

    Test case with this decorator is automatically activated for testing
    if the specified class can be successfully imported,
    otherwise it is not tested but just leaves a warning.

    This is useful when the version of Chainer the current environment has does
    not suppor the operator that the test case tests.
    For example, Chaienr v3 doesn't have Shift operator, so the test case for
    Shift has `@require_import('chainer.functions.connections.shift.Shift')`
    decorator.

    Args:
        func_fqn: Fully qualified name of Chainer function class.
          For example, "chainer.functions.activation.relu.ReLU".
    """
    def nothing(tester_func):
        def nothing_body(*args):
            msg = "The test case \"{}\" requires \"{}\" but it cannot be " \
                  "imported. Skipping".format(func_fqn, tester_func.__name__)
            warnings.warn(msg)
            return None
        return nothing_body

    def run_test(tester_func):
        def f(*args):
            return tester_func(*args)
        return f

    try:
        func_module, func_class = func_fqn.rsplit('.', 1)
        m = importlib.import_module(func_module)
        func = getattr(m, func_class)
    except ImportError:
        func = None
    except AttributeError:
        func = None

    if func is None:
        return nothing
    return run_test


def require_chainer_version(ver_oldest, ver_newest=None):
    """Decorator to turn on/off a test case by Chainer version

    Test case with this decorator is automatically activated for testing
    if the current Chainer version is between `ver_oldest` and `ver_newest`.
    In case `ver_newest` is not specified it is ignored.

    This is useful when the version of Chainer the current environment has does
    not suppor the operator that the test case tests.
    For example, Chaienr v3 doesn't have `groups` argument in Convolution2D.
    It cannot be checked by @require_import decorator so in this case
    this making use of this decorator like `@require_chainer_version('4.0.0')`
    would be appropriate.

    Args:
        ver_oldest: Version string of lower bound (inclusive).
          For example, '3.0.0'.
        ver_newest: Version string of upper bound (inclusive). Can be omitted.
    """
    def nothing(tester_func):
        def nothing_body(*args):
            ver_msg = "newer than or equals to {}".format(ver_oldest)
            if ver_newest is not None:
                ver_msg += " and older than or equal to {}".format(ver_newest)
            msg = "The test case \"{}\" requires Chainer version to be {}. "\
                  "Actual version is {}. Skipping."\
                  .format(tester_func.__name__, ver_msg, chainer.__version__)
            warnings.warn(msg)
            return None
        return nothing_body

    def run_test(tester_func):
        def f(*args):
            return tester_func(*args)
        return f

    ver_current = LooseVersion(chainer.__version__)
    if LooseVersion(ver_oldest) <= ver_current:
        if ver_newest is None or ver_current <= LooseVersion(ver_newest):
            return run_test
    return nothing
