"""Functions for easy interactions with IPython and IPython notebooks.

NotebookRunner is modified from runipy.
This modified code is included under the terms of its license:

Copyright (c) 2013, Paul Butler
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import absolute_import

import os
import platform
import re
import sys
import time
import unicodedata

try:
    from IPython import get_ipython
    from IPython.display import HTML
    from IPython.nbconvert import HTMLExporter, PythonExporter
    from IPython.nbformat import current
except ImportError:
    def get_ipython():
        return None

import numpy as np


def in_ipynb():
    """Determines if code is executed in an IPython notebook.

    Returns
    -------
    bool
       ``True`` if the code is executed in an IPython notebook, otherwise
       ``False``.

    Notes
    -----
    It is possible to connect to a kernel started from an IPython notebook
    from outside of the notebook. Thus, this function might return ``True``
    even though the code is not running in an IPython notebook.
    """
    if get_ipython() is not None:
        cfg = get_ipython().config
        app_key = 'IPKernelApp'
        if 'parent_appname' not in cfg[app_key]:
            app_key = 'KernelApp'  # was used by old IPython versions
        if cfg[app_key].get('parent_appname') == 'ipython-notebook':
            return True
    return False


def has_ipynb_widgets():
    """Determines whether IPython widgets are available.

    Returns
    -------
    bool
        ``True`` if IPython widgets are available, otherwise ``False``.
    """
    try:
        import IPython.html.widgets
        import IPython.utils.traitlets
        assert IPython.html.widgets
        assert IPython.utils.traitlets
    except ImportError:
        return False
    else:
        return True


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


def load_notebook(nb_path):
    with open(nb_path) as f:
        nb = current.read(f, 'json')
    return nb


def export_py(nb, dest_path=None):
    """Convert notebook to Python script.

    Optionally saves script to dest_path.
    """
    exporter = PythonExporter()
    body, resources = exporter.from_notebook_node(nb)
    if sys.version_info[0] == 2:
        body = unicodedata.normalize('NFKD', body).encode('ascii', 'ignore')
    # We'll remove %matplotlib inline magic, but leave the rest
    body = body.replace("get_ipython().magic(u'matplotlib inline')\n", "")
    body = body.replace("get_ipython().magic('matplotlib inline')\n", "")
    if dest_path is not None:
        with open(dest_path, 'w') as f:
            f.write(body)
    return body


def export_html(nb, dest_path=None):
    """Convert notebook to HTML.

    Optionally saves HTML to dest_path.
    """
    exporter = HTMLExporter(template_file='full')
    output, resources = exporter.from_notebook_node(nb)
    header = output.split('<head>', 1)[1].split('</head>', 1)[0]
    body = output.split('<body>', 1)[1].split('</body>', 1)[0]

    # Monkeypatch CSS
    header = header.replace('<style', '<style scoped="scoped"')
    header = header.replace(
        'body {\n  overflow: visible;\n  padding: 8px;\n}\n', '')
    header = header.replace("code,pre{", "code{")

    # Filter out styles that conflict with the sphinx theme.
    bad_anywhere = ['navbar',
                    'body{',
                    'alert{',
                    'uneditable-input{',
                    'collapse{']
    bad_anywhere.extend(['h%s{' % (i+1) for i in range(6)])

    bad_beginning = ['pre{', 'p{margin']

    header_lines = [x for x in header.split('\n')
                    if (not any(x.startswith(s) for s in bad_beginning)
                        and not any(s in x for s in bad_anywhere))]
    header = '\n'.join(header_lines)

    # Concatenate raw html lines
    lines = ['<div class="ipynotebook">']
    lines.append(header)
    lines.append(body)
    lines.append('</div>')
    html_out = '\n'.join(lines)
    if dest_path is not None:
        with open(dest_path, 'w') as f:
            f.write(html_out)
    return html_out


def export_evaluated(nb, dest_path=None, skip_exceptions=False):
    """Convert notebook to an evaluated notebook.

    Optionally saves the notebook to dest_path.
    """
    nb_runner = NotebookRunner(nb)
    try:
        nb_runner.run_notebook(skip_exceptions=skip_exceptions)
    except Exception as e:
        return ("Notebook conversion failed with the following traceback: \n%s"
                % re.sub(r'\\033[\[\]]([0-9]{1,2}([;@][0-9]{0,2})*)*[mKP]?',
                         '', str(e)))
    if dest_path is not None:
        with open(dest_path, 'w') as f:
            current.write(nb_runner.nb, f, 'json')
    return nb_runner.nb


class NotebookRunner(object):
    # The kernel communicates with mime-types while the notebook
    # uses short labels for different cell types. We'll use this to
    # map from kernel types to notebook format types.

    MIME_MAP = {
        'image/jpeg': 'jpeg',
        'image/png': 'png',
        'text/plain': 'text',
        'text/html': 'html',
        'text/latex': 'latex',
        'application/javascript': 'html',
        'image/svg+xml': 'svg',
    }

    def __init__(self, nb, working_dir=None):
        from IPython.kernel import KernelManager

        self.km = KernelManager()

        cwd = os.getcwd()
        if working_dir is not None:
            os.chdir(working_dir)
        self.km.start_kernel()
        os.chdir(cwd)

        if platform.system() == 'Darwin':
            # There is sometimes a race condition where the first
            # execute command hits the kernel before it's ready.
            # It appears to happen only on Darwin (Mac OS) and an
            # easy (but clumsy) way to mitigate it is to sleep
            # for a second.
            time.sleep(1)

        self.kc = self.km.client()
        self.kc.start_channels()
        self.shell = self.kc.shell_channel
        self.iopub = self.kc.iopub_channel
        self.nb = nb

    def __del__(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)

    def run_cell(self, cell):
        """Run a notebook cell and update the output of that cell in-place."""
        self.shell.execute(cell.input)
        reply = self.shell.get_msg()
        status = reply['content']['status']
        if status == 'error':
            traceback_text = ("Cell raised uncaught exception: \n"
                              "\n".join(reply['content']['traceback']))

        outs = []
        while True:
            msg = self.iopub.get_msg(timeout=1)
            msg_type = msg['msg_type']
            content = msg['content']

            if msg_type == 'status' and content['execution_state'] == 'idle':
                break

            # IPython 3.0.0-dev writes pyerr/pyout in the notebook format
            # but uses error/execute_result in the message spec. This
            # the translation needed for tests to pass with IPython 3.0.0-dev
            notebook3_format_conversions = {
                'error': 'pyerr',
                'execute_result': 'pyout',
            }
            msg_type = notebook3_format_conversions.get(msg_type, msg_type)

            out = current.NotebookNode(output_type=msg_type)

            if 'execution_count' in content:
                cell['prompt_number'] = content['execution_count']
                out.prompt_number = content['execution_count']

            if msg_type in ('status', 'pyin', 'execute_input'):
                continue
            elif msg_type == 'stream':
                out.stream = content['name']
                out.text = content['data']
            elif msg_type in ('display_data', 'pyout'):
                for mime, data in content['data'].items():
                    try:
                        attr = self.MIME_MAP[mime]
                    except KeyError:
                        raise NotImplementedError(
                            'unhandled mime type: %s' % mime)
                    setattr(out, attr, data)
            elif msg_type == 'pyerr':
                out.ename = content['ename']
                out.evalue = content['evalue']
                out.traceback = content['traceback']
            elif msg_type == 'clear_output':
                outs = []
                continue
            else:
                raise NotImplementedError(
                    'unhandled iopub message: %s' % msg_type)
            outs.append(out)
        cell['outputs'] = outs

        if status == 'error':
            raise Exception(traceback_text)

    def iter_code_cells(self):
        """Iterate over the notebook cells containing code."""
        for ws in self.nb.worksheets:
            for cell in ws.cells:
                if cell.cell_type == 'code':
                    yield cell

    def run_notebook(self, skip_exceptions=False):
        """Runs all notebook cells in order and updates outputs in-place.

        If ``skip_exceptions`` is True, then if exceptions occur in a cell, the
        subsequent cells are run (by default, the notebook execution stops).
        """
        for cell in self.iter_code_cells():
            try:
                self.run_cell(cell)
            except:
                if not skip_exceptions:
                    raise
