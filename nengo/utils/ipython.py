from __future__ import absolute_import

import sys
import unicodedata

from IPython.display import HTML
from IPython.nbconvert import PythonExporter
from IPython.nbformat import current
import numpy as np


def hide_input():
    """Hide the input of the IPython notebook input block this is executed in.

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
                while (cell.className.split(' ')[0] != "cell") {
                    cell = cell.parentNode;
                }
                var input_%(uuid)s;
                for (var i = 0; i < cell.children.length; i++) {
                    if (cell.children[i].className.split(' ')[0] == "input")
                        input_%(uuid)s = cell.children[i];
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
                var input_%(uuid)s = cell_%(uuid)s.children("div.input");
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


def export_py(nb, dst_path):
    export = PythonExporter()
    body, resources = export.from_notebook_node(nb)
    if sys.version_info[0] == 2:
        body = unicodedata.normalize('NFKD', body).encode('ascii', 'ignore')
    # We'll remove %matplotlib inline magic, but leave the rest
    body = body.replace("get_ipython().magic(u'matplotlib inline')\n", "")
    body = body.replace("get_ipython().magic('matplotlib inline')\n", "")
    with open(dst_path, 'w') as f:
        f.write(body)


def load_notebook(nb_path):
    with open(nb_path) as f:
        nb = current.reads(f.read(), 'json')
    return nb
