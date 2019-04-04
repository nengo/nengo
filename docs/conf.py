# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set
# to its containing dir.

import os
import shutil
import sys

try:
    import nengo
    import nengo_sphinx_theme  # noqa: F401 pylint: disable=unused-import
except ImportError:
    print("To build the documentation, nengo and nengo_sphinx_theme must be "
          "installed in the current environment. Please install these and "
          "their requirements first. A virtualenv is recommended!")
    sys.exit(1)


def copy_examples(app, exception):
    if exception is None:
        download_path = os.path.join(app.outdir, '_downloads')
        dest_path = os.path.join(download_path, 'examples')
        if not os.path.exists(download_path):
            os.mkdir(download_path)
        if os.path.exists(dest_path):
            shutil.rmtree(dest_path)
        shutil.copytree(os.path.join(app.srcdir, 'examples'), dest_path)


def setup(app):
    app.connect('build-finished', copy_examples)


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'nengo_sphinx_theme',
    'nbsphinx',
    'numpydoc',
]

# -- sphinx.ext.autodoc
autoclass_content = 'both'  # class and __init__ docstrings are concatenated
autodoc_default_options = {"members": None}
autodoc_member_order = 'bysource'  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sklearn': ('https://scikit-learn.org/dev', None),
}

# -- sphinx.ext.todo
todo_include_todos = True

# -- numpydoc config
numpydoc_show_class_members = False

# -- nbsphinx
nbsphinx_timeout = -1

# -- sphinx
needs_sphinx = '1.3'
nitpicky = True
exclude_patterns = ['_build', '**/.ipynb_checkpoints']
linkcheck_timeout = 30
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

project = 'Nengo'
authors = 'Applied Brain Research'
copyright = nengo.__copyright__
version = '.'.join(nengo.__version__.split('.')[:2])  # Short X.Y version
release = nengo.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

pygments_style = "sphinx"
templates_path = ["_templates"]
html_static_path = ["_static"]

html_theme = "nengo_sphinx_theme"

html_title = "Nengo core {0} docs".format(release)
htmlhelp_basename = 'Nengo core'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False
html_favicon = os.path.join("_static", "favicon.ico")
html_theme_options = {
    "sidebar_logo_width": 200,
    "nengo_logo": "general-full-light.svg",
}

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
    # 'preamble': '',
}

latex_documents = [
    # (source start file, target, title, author, documentclass [howto/manual])
    ('index', 'nengo.tex', html_title, authors, 'manual'),
]

# -- Options for manual page output -------------------------------------------

man_pages = [
    # (source start file, name, description, authors, manual section).
    ('index', 'nengo', html_title, [authors], 1)
]

# -- Options for Texinfo output -----------------------------------------------

texinfo_documents = [
    # (source start file, target, title, author, dir menu entry,
    #  description, category)
    ('index', 'nengo', html_title, authors, 'Nengo',
     'Large-scale neural simulation in Python', 'Miscellaneous'),
]
