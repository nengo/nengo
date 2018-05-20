"""Functions for easy interactions with IPython and IPython notebooks."""

from __future__ import absolute_import

import io

import numpy as np

try:
    import IPython
    from IPython import get_ipython
    from IPython.display import HTML

    if IPython.version_info[0] <= 3:
        from IPython.nbconvert import PythonExporter
    else:
        from nbconvert import PythonExporter

    if IPython.version_info[0] <= 3:
        # pylint: disable=ungrouped-imports
        from IPython import nbformat
    else:
        import nbformat

except ImportError:
    def get_ipython():
        return None
assert get_ipython


def check_ipy_version(min_version):
    try:
        import IPython
        return IPython.version_info >= min_version
    except ImportError:
        return False


def hide_input():
    """Hide the input of the Jupyter notebook input block this is executed in.

    Returns a link to toggle the visibility of the input block.
    """
    uuid = np.random.randint(np.iinfo(np.int32).max)

    script = """
        <a id="%(uuid)s" href="javascript:toggle_input_%(uuid)s()"
          >Show Input</a>

        <script type="text/javascript">
        var toggle_input_%(uuid)s;
        (function() {
            if (typeof jQuery == 'undefined') {
                // no jQuery
                var link_%(uuid)s = document.getElementById("%(uuid)s");
                var cell = link_%(uuid)s;
                while (cell.className.split(' ')[0] != "cell"
                       && cell.className.split(' ')[0] != "nboutput") {
                    cell = cell.parentNode;
                }
                var input_%(uuid)s;
                if (cell.className.split(' ')[0] == "cell") {
                    for (var i = 0; i < cell.children.length; i++) {
                        if (cell.children[i].className.split(' ')[0]
                            == "input") {
                            input_%(uuid)s = cell.children[i];
                        }
                    }
                } else {
                    input_%(uuid)s = cell.previousElementSibling;
                }
                input_%(uuid)s.style.display = "none"; // hide

                toggle_input_%(uuid)s = function() {
                    if (input_%(uuid)s.style.display == "none") {
                        input_%(uuid)s.style.display = ""; // show
                        link_%(uuid)s.innerHTML = "Hide Input";
                    } else {
                        input_%(uuid)s.style.display = "none"; // hide
                        link_%(uuid)s.innerHTML = "Show Input";
                    }
                }

            } else {
                // jQuery
                var link_%(uuid)s = $("a[id='%(uuid)s']");
                var cell_%(uuid)s = link_%(uuid)s.parents("div.cell:first");
                if (cell_%(uuid)s.length == 0) {
                    cell_%(uuid)s = link_%(uuid)s.parents(
                        "div.nboutput:first");
                }
                var input_%(uuid)s = cell_%(uuid)s.children("div.input");
                if (input_%(uuid)s.length == 0) {
                    input_%(uuid)s = cell_%(uuid)s.prev("div.nbinput");
                }
                input_%(uuid)s.hide();

                toggle_input_%(uuid)s = function() {
                    if (input_%(uuid)s.is(':hidden')) {
                        input_%(uuid)s.slideDown();
                        link_%(uuid)s[0].innerHTML = "Hide Input";
                    } else {
                        input_%(uuid)s.slideUp();
                        link_%(uuid)s[0].innerHTML = "Show Input";
                    }
                }
            }
        }());
        </script>
    """ % dict(uuid=uuid)

    return HTML(script)


def load_notebook(nb_path):
    with io.open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def export_py(nb, dest_path=None):
    """Convert notebook to Python script.

    Optionally saves script to dest_path.
    """
    exporter = PythonExporter()
    body, resources = exporter.from_notebook_node(nb)

    # Remove all lines with get_ipython
    while u"get_ipython()" in body:
        ind0 = body.find(u"get_ipython()")
        ind1 = body.find(u"\n", ind0)
        body = body[:ind0] + body[(ind1 + 1):]

    if u"plt" in body:
        body += u"\nplt.show()\n"

    if dest_path is not None:
        with io.open(dest_path, 'w', encoding='utf-8') as f:
            f.write(body)
    return body


def iter_cells(nb, cell_type="code"):
    return (cell for cell in nb.cells if cell.cell_type == cell_type)
