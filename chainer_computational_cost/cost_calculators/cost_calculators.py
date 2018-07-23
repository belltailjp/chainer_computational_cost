import inspect
import sys
import warnings


calculators = dict()


def check_signature(func):
    """Check cost calculator's signature

    Cost calculator has to have the following parameter.
    - func
    - in_data
    - **kwargs
    Name can be different.
    """
    if not callable(func):
        return False

    if sys.version_info < (3,):
        p = inspect.getargspec(func)
        if len(p.args) != 2 or p.varargs is not None or p.keywords is None:
            return False
        return True
    else:
        p = inspect.signature(func).parameters
        if len(p) != 3:
            return False

        _, _, kwargs = p.keys()
        if p[kwargs].kind != inspect.Parameter.VAR_KEYWORD:
            return False

        return True

    return True


def register(func):
    """A decorator to register cost calculator function (internal use only)

    By specifying this decorator to a cost calculator function,
    chainer-computational-cost recognizes that function.
    """
    def reg(calculator):
        if not check_signature(calculator):
            warnings.warn("cost calculator signature mismatch: {}"
                          .format(func.__name__))
        else:
            calculators[func] = calculator

        def f(*args, **kwargs):
            return calculator(*args, **kwargs)
        return f
    return reg
