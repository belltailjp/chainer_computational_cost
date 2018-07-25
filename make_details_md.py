import inspect
import os
import re
import textwrap
import urllib
import warnings

from chainer_computational_cost.cost_calculators import calculators


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


if __name__ == "__main__":
    contents = []
    table_of_contents = ["# Table of contents", '']

    chapters = set()
    for t, f in calculators.items():
        if f.__doc__ is None:
            continue

        # "/path/to/activation.py" -> "Activation"
        src_file = os.path.basename(inspect.getfile(f))
        module_name = os.path.splitext(src_file)[0]
        chapter = module_name.replace('_', ' ').capitalize()

        # Print h1 heading if this module appears first
        if chapter not in chapters:
            contents.append("# {}\n".format(chapter))
            chapters.add(chapter)
            fmt = "* [{}](#{})".format(chapter, module_name.replace('_', '-'))
            table_of_contents.append(fmt)

        # Format docstring content (remove indent, replace equations)
        ds = f.__doc__.splitlines()
        func_name, ds = ds[0], "\n".join(ds[1:])
        ds = textwrap.dedent(ds).strip()
        ds = re.sub(r'\$\$.+\$\$', eq_to_url, ds, flags=re.MULTILINE)
        ds = re.sub(r'\$.+?\$', inline_eq_to_url, ds)

        contents.append("## {}\n".format(func_name))
        contents.append("{}\n\n".format(ds))

        # h2 should be same as chainer's function eg PReLUFunction
        # so that it can automatically generate anchor links
        if t.__name__ not in func_name:
            warnings.error("docstring header \"{}\" doesn't contain full"
                           "function name {}".format(func_name, t.__name__))
        fmt = "  * [{}](#{})".format(t.__name__, t.__name__.lower())
        table_of_contents.append(fmt)

    with open("DETAILS.md", "wt") as f:
        f.write("\n".join(table_of_contents))
        f.write("\n" * 3)
        f.write("\n".join(contents))
