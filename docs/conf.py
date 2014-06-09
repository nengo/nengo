# -*- coding: utf-8 -*-
#
# This file is execfile()d with the current directory set
# to its containing dir.

import sys

try:
    import nengo
    import nengo_sphinx_theme
except ImportError:
    print ("To build the documentation, nengo and nengo_sphinx_theme must be "
           "installed in the current environment. Please install these and "
           "their requirements first. A virtualenv is recommended!")
    sys.exit(1)

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'numpydoc',
    'nengo.utils.docutils',
]

# -- sphinx.ext.todo config
todo_include_todos = True
# -- numpydoc config
numpydoc_show_class_members = False

# -- sphinx config
exclude_patterns = ['_build']
source_suffix = '.rst'
source_encoding = 'utf-8'
master_doc = 'index'

project = u'Nengo'
authors = u'Applied Brain Research'
copyright = nengo.__copyright__
version = '.'.join(nengo.__version__.split('.')[:2])  # Short X.Y version
release = nengo.__version__  # Full version, with tags
pygments_style = 'default'

# -- Options for HTML output --------------------------------------------------

html_theme = 'nengo_sphinx_theme'
html_theme_path = [nengo_sphinx_theme.get_html_theme_path()]
html_title = "Nengo {0} docs".format(release)
# html_static_path = ['_static']
html_use_smartypants = True
htmlhelp_basename = 'Nengodoc'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
html_show_sphinx = False

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
