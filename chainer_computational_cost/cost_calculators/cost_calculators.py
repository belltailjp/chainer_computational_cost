from collections import OrderedDict
import importlib
import inspect
import six
import warnings

calculators = OrderedDict()     # active calculators

# all the calculators including those cannot be activated
# (not disclosed to outside, but used by make_details_md.py)
all_calculators = OrderedDict()


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

    if six.PY2:
        p = inspect.getargspec(func)
        if len(p.args) != 2 or p.varargs is not None or p.keywords is None:
            return False
    else:
        p = inspect.signature(func).parameters
        if len(p) != 3:
            return False

        _, _, kwargs = p.keys()
        if p[kwargs].kind != inspect.Parameter.VAR_KEYWORD:
            return False

    return True


def register(func):
    """A decorator to register cost calculator function (internal use only)

    This registers the function as a cost calculator function for the specified
    type of Chainer Function.
    You can specify the target Chainer Function by the following ways.

    (1) Type of Chainer Function (FunctionNode)
    You can directly pass the type object to the decorator.
    If the type may not exist in some Chainer versions, try the second way.

    (2) Fully qualified name of a Chainer Function.
    chainer-computational-cost tries to import it and registers the cost
    calculator for the Function.
    In case the specified Chainer Function is not found, for example the
    current chainer version doesn't support that Function yet,
    the cost calculator will not be registered.
    For example, `"chainer.functions.activation.relu.ReLU"`

    args:
      func: Chainer Function that you want the cost calculator function to be
        registered for.
    """
    if type(func) is str:
        func_name = func
        try:
            # F.activation.relu.ReLU -> ['F.activation.relu', 'ReLU']
            func_module, func_class = func.rsplit('.', 1)
            m = importlib.import_module(func_module)
            func = getattr(m, func_class)
        except ImportError:
            func = None
        except AttributeError:
            func = None
    else:
        func_name = func.__name__

    def reg(calculator):
        if not check_signature(calculator):
            warnings.warn("cost calculator signature mismatch: {}"
                          .format(func_name))
        elif func is not None:
            # If the function exists
            calculators[func] = calculator
            all_calculators[func] = calculator
        else:
            # register all the defined calculators including those cannot be
            # activated (e.g. chainer in this env is too old)
            all_calculators[func_name] = calculator
    return reg
