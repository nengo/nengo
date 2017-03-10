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

import errno
import logging
import os
import shutil
import tempfile
import warnings

from docutils import nodes
from docutils.parsers.rst import Directive, directives

from . import ipython as ipext

logger = logging.getLogger(__name__)


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
                "'{}'".format(nb_path))

        # Check if raw HTML is supported
        if not self.state.document.settings.raw_enabled:
            raise self.warning('"%s" directive disabled.' % self.name)

        cwd = os.getcwd()
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)

        # Get path to notebook
        nb_filename = self.arguments[0]
        nb_basename = os.path.basename(nb_filename)
        rst_file = self.state_machine.document.attributes['source']
        rst_dir = os.path.abspath(os.path.dirname(rst_file))
        nb_abs_path = os.path.abspath(os.path.join(rst_dir, nb_filename))

        # Move files around
        rel_dir = os.path.relpath(rst_dir, setup.confdir)
        dest_dir = os.path.join(setup.app.builder.outdir, rel_dir)
        dest_path = os.path.join(dest_dir, nb_basename)

        image_dir, image_rel_dir = make_image_dir(setup, rst_dir)

        # Ensure destination build directory exists
        safe_mkdir(os.path.dirname(dest_path))

        # Construct paths to versions getting copied over
        dest_path_script = dest_path.replace('.ipynb', '.py')
        rel_path_py = nb_basename.replace('.ipynb', '.py')

        # Create python script version (if necessary)
        nb = ipext.load_notebook(nb_abs_path)
        ipext.export_py(nb, dest_path_script)

        # Create evaluated ipynb (if necessary)
        skip_exceptions = 'skip_exceptions' in self.options
        try:
            nb = ipext.export_evaluated(
                nb, dest_path, skip_exceptions=skip_exceptions)
        except Exception as e:
            warnings.warn("Conversion of %s failed with the following "
                          "traceback:\n%s" % (nb_filename, e))
        evaluated_html = ipext.export_html(
            nb, image_dir=image_dir, image_rel_dir=image_rel_dir)

        # Create link to notebook and script files
        link_rst = "Download ``%s`` as an %s or %s." % (
            os.path.splitext(nb_basename)[0],
            formatted_link(nb_basename, "IPython notebook"),
            formatted_link(rel_path_py, "Python script"))

        self.state_machine.insert_input([link_rst], rst_file)

        # Create notebook node
        nb_node = NotebookNode(
            '', evaluated_html, format='html', source=nb_path)
        nb_node.source, nb_node.line = (
            self.state_machine.get_source_and_line(self.lineno))

        # Add dependency
        self.state.document.settings.record_dependencies.add(nb_abs_path)

        # Clean up
        os.chdir(cwd)
        shutil.rmtree(tmpdir, True)

        return [nb_node]


def make_image_dir(setup, rst_dir):
    image_dir = os.path.join(setup.app.builder.outdir, '_images')
    rel_dir = os.path.relpath(setup.confdir, rst_dir)
    image_rel_dir = os.path.join(rel_dir, '_images')
    safe_mkdir(image_dir)
    return image_dir, image_rel_dir


def safe_mkdir(dirname):
    try:
        os.makedirs(dirname)
    except OSError as err:
        if err.errno != errno.EEXIST:
            logger.warning("OSError during safe_mkdir: %s", err)


def formatted_link(path, text=None):
    if text is None:
        text = os.path.basename(path)
    return "`%s <%s>`__" % (text, path)


class NotebookNode(nodes.raw):
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
    app.add_node(NotebookNode,
                 html=(visit_notebook_node, depart_notebook_node))
    app.add_directive('notebook', NotebookDirective)
