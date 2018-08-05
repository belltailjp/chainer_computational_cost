import inspect
import os
import re
import six.moves.urllib as urllib
import textwrap
import warnings

from chainer_computational_cost.cost_calculators.cost_calculators import \
    all_calculators

head = """
    # Details of theoretical computational costs

    This document explains how chainer-computational-cost estimates
    theoretical computational cost of each type of layer.
    Unless otherwise specified, $x$ stands for the input to the layer and
    $y$ is output. $ \| x \| $ is the number of elements in $x$
    (equivalent to `x.size`), if $x$ is empty or does not exist,
    $\| x \| = 0$.

    The basic strategy of how the "theoretical computational cost" is defined
    is written in [README.md](README.md).
    """


def inline_eq_to_url(s):
    s = s.group().replace('$', '')
    s = "\\dpi{100} \\normal " + s
    s = urllib.parse.quote(s)
    base_url = "https://latex.codecogs.com/png.latex?"
    return "<img src=\"{}{}\"/>".format(base_url, s)


def eq_to_url(s):
    s = s.group().replace('$', '')
    s = "\\dpi{130} \\normal " + s
    s = urllib.parse.quote(s)
    base_url = "https://latex.codecogs.com/png.latex?"
    return "<img src=\"{}{}\"/>".format(base_url, s)


def format_content(content):
    content = textwrap.dedent(content).strip()
    content = re.sub(r'\$\$.+\$\$', eq_to_url, content, flags=re.MULTILINE)
    content = re.sub(r'\$.+?\$', inline_eq_to_url, content)
    return content


if __name__ == "__main__":
    contents = []
    table_of_contents = ["## Table of contents", '']

    chapters = set()
    for t, f in all_calculators.items():
        if f.__doc__ is None:
            continue

        # in most cases t is type object of a FunctionNode class,
        # but when the cost calculator cannot be registered,
        # it can be a string of fully qualified class name of a FunctionNode.
        # (see how @register decorator makes all_calculators)
        if type(t) is str:
            t = t.split('.')[-1]    # F.activation.relu.ReLU -> ReLU
        else:
            t = t.__name__

        # "/path/to/activation.py" -> "Activation"
        src_file = os.path.basename(inspect.getfile(f))
        module_name = os.path.splitext(src_file)[0]
        chapter = module_name.replace('_', ' ').capitalize()

        # Print h1 heading if this module appears first
        if chapter not in chapters:
            contents.append("## {}\n".format(chapter))
            chapters.add(chapter)
            fmt = "* [{}](#{})".format(chapter, module_name.replace('_', '-'))
            table_of_contents.append(fmt)

        # Format docstring content (remove indent, replace equations)
        ds = f.__doc__.splitlines()
        func_name, ds = ds[0], "\n".join(ds[1:])
        ds = format_content(textwrap.dedent(ds).strip())

        contents.append("### {}\n".format(func_name))
        contents.append("{}\n\n".format(ds))

        # h2 should be same as chainer's function eg PReLUFunction
        # so that it can automatically generate anchor links
        if t not in func_name:
            warnings.error("docstring header \"{}\" doesn't contain full"
                           "function name {}".format(func_name, t))
        fmt = "  * [{}](#{})".format(t, t.lower())
        table_of_contents.append(fmt)

    with open("DETAILS.md", "wt") as f:
        f.write(format_content(head))
        f.write("\n" * 3)
        f.write("\n".join(table_of_contents))
        f.write("\n" * 3)
        f.write("\n".join(contents))
