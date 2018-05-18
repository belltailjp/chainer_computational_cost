import inspect
import os
import re
import textwrap
import urllib

import chainer_computational_cost.cost_calculators


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
    out_f = open("DETAILS.md", "wt")

    chapters = set()
    for f in chainer_computational_cost.cost_calculators.calculators.values():
        if f.__doc__ is None:
            continue

        # "/path/to/activation.py" -> "Activation"
        src_file = os.path.basename(inspect.getfile(f))
        chapter = os.path.splitext(src_file)[0].capitalize()
        if chapter not in chapters:
            out_f.write("# {}\n\n".format(chapter))
            chapters.add(chapter)

        # format
        ds = f.__doc__.splitlines()
        func_name, ds = ds[0], "\n".join(ds[1:])
        ds = textwrap.dedent(ds).strip()
        ds = re.sub(r'\$\$.+\$\$', eq_to_url, ds, flags=re.MULTILINE)
        ds = re.sub(r'\$.+?\$', inline_eq_to_url, ds)

        out_f.write("## {}\n\n".format(func_name))
        out_f.write("{}\n\n\n".format(ds))
