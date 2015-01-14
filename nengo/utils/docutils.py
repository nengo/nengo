"""reStructuredText directives used to build Nengo documentation.

The functions in this file were modified from RunNotebook.
This modified code is included under the terms of its license:

Copyright (c) 2013 Nathan Goldbaum. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

from __future__ import absolute_import

import os
import warnings

from docutils import nodes
from docutils.parsers.rst import Directive, directives

import nengo.utils.ipython as ipext


class NotebookDirective(Directive):
    """Insert an evaluated notebook into a document.

    This runs the notebook and uses nbconvert to transform a path to an
    unevaluated notebook into HTML suitable for embedding in a Sphinx document.
    """
    required_arguments = 1
    optional_arguments = 1
    option_spec = {'skip_exceptions': directives.flag}
    final_argument_whitespace = True

    def run(self):
        nb_path = self.arguments[0]

        if ' ' in nb_path:
            raise ValueError(
                "Due to issues with docutils stripping spaces from links, "
                "white space is not allowed in notebook filenames: "
                "'{0}'".format(nb_path))

        # Check if raw HTML is supported
        if not self.state.document.settings.raw_enabled:
            raise self.warning('"%s" directive disabled.' % self.name)

        # Get path to notebook
        source_dir = os.path.dirname(
            os.path.abspath(self.state.document.current_source))
        nb_filename = self.arguments[0]
        nb_basename = os.path.basename(nb_filename)
        rst_file = self.state_machine.document.attributes['source']
        rst_dir = os.path.abspath(os.path.dirname(rst_file))
        nb_abs_path = os.path.abspath(os.path.join(rst_dir, nb_filename))

        # Move files around
        rel_dir = os.path.relpath(rst_dir, setup.confdir)
        rel_path = os.path.join(rel_dir, nb_basename)
        dest_dir = os.path.join(setup.app.builder.outdir, rel_dir)
        dest_path = os.path.join(dest_dir, nb_basename)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        dest_path_py = dest_path.replace('.ipynb', '.py')
        rel_path_py = nb_basename.replace('.ipynb', '.py')

        nb = ipext.load_notebook(nb_abs_path)
        skip_exceptions = 'skip_exceptions' in self.options

        # Create python script version (if necessary)
        ipext.export_py(nb, dest_path_py)

        # Create evaluated ipynb (if necessary)
        try:
            nb_eval = ipext.export_evaluated(
                nb, dest_path, skip_exceptions=skip_exceptions)
        except Exception as e:
            warnings.warn(
                "Notebook conversion failed with the following traceback: \n"
                + str(e))
            nb_eval = nb
        evaluated_html = ipext.export_html(nb_eval)

        # Create link to notebook and script files
        link_rst = "Download ``%s`` as an %s or %s." % (
            os.path.splitext(nb_basename)[0],
            formatted_link(nb_basename, "IPython notebook"),
            formatted_link(rel_path_py, "Python script"))

        self.state_machine.insert_input([link_rst], rst_file)

        # Create notebook node
        nb_node = notebook_node(
            'test', evaluated_html, format='html', source=nb_path)
        nb_node.source, nb_node.line = (
            self.state_machine.get_source_and_line(self.lineno))

        # Add dependency
        self.state.document.settings.record_dependencies.add(nb_abs_path)

        return [nb_node]


def formatted_link(path, text=None):
    if text is None:
        text = os.path.basename(path)
    return "`%s <%s>`__" % (text, path)


class notebook_node(nodes.raw):
    """An evaluated IPython notebook"""


def visit_notebook_node(self, node):
    self.visit_raw(node)


def depart_notebook_node(self, node):
    self.depart_raw(node)


def setup(app):
    """Let Sphinx know about the Notebook directive.

    When Sphinx imports this module, it will run this function.
    We add our directives here so that we can use them in our docs.
    """
    setup.app = app
    setup.config = app.config
    setup.confdir = app.confdir
    app.add_node(notebook_node,
                 html=(visit_notebook_node, depart_notebook_node))
    app.add_directive('notebook', NotebookDirective)
