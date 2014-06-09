# -*- coding: utf-8 -*-

import sys

import nengo_sphinx_theme

import nengo.version

# -- General configuration ----------------------------------------------------

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

# needs_sphinx = '1.0'
# templates_path = ['_templates']
source_suffix = '.rst'
# source_encoding = 'utf-8-sig'
master_doc = 'index'
project = u'Nengo'
copyright = u'2013, CNRGlab @ UWaterloo'
version = '.'.join(str(v) for v in nengo.version.version_info[:2])
release = nengo.version.version
exclude_patterns = ['_build']
# add_function_parentheses = True
# add_module_names = True
# show_authors = False
pygments_style = 'default'
# mod index_common_prefix = []

# -- Options for HTML output --------------------------------------------------

html_theme = 'nengo_sphinx_theme'
html_theme_path = [nengo_sphinx_theme.get_html_theme_path()]
# html_theme_options = {
# }

# html_title = None  # "<project> v<release> documentation"
# html_short_title = None  # html_title
# html_logo = None
# html_favicon = None
# html_static_path = ['_static']
# html_last_updated_fmt = '%b %d, %Y'
html_last_updated_fmt = ''  # Suppress 'Last updated on:' timestamp
# html_use_smartypants = True
# html_sidebars = {}
# html_additional_pages = {}
# html_domain_indices = True
# html_use_index = True
# html_split_index = False
# html_show_sourcelink = True
html_show_sphinx = False
# html_show_copyright = True
# html_use_opensearch = ''
# html_file_suffix = None
htmlhelp_basename = 'nengodoc'

# -- Options for LaTeX output -------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual])
latex_documents = [
    ('index', 'nengo.tex', u'Nengo Documentation',
     u'CNRGlab @ UWaterloo', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True


# -- Options for manual page output -------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    ('index', 'nengo', u'Nengo Documentation',
     [u'CNRGlab @ UWaterloo'], 1)
]

# If true, show URL addresses after external links.
#man_show_urls = False


# -- Options for Texinfo output -----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    ('index', 'nengo', u'Nengo Documentation',
     u'CNRGlab @ UWaterloo', 'Nengo',
     'A Python library for building and simulating large-scale brain models',
     'Miscellaneous'),
]

# Documents to append as an appendix to all manuals.
#texinfo_appendices = []

# If false, no module index is generated.
#texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
#texinfo_show_urls = 'footnote'
