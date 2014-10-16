***************
Developer Guide
***************

The following sections will help you change
how Nengo builds and simulates brain models.
If you make Nengo do something cool,
we hope that you'll consider contributing
to Nengo development!

Let's start off with some basics
in case you missed them in the README.

How to Build Nengo
==================

.. code-block:: bash

   git clone git@github.com:ctn-waterloo/nengo-temp.git
   cd nengo-temp
   python setup.py develop

How to Build the Documentation
==============================

Note that you can only build the documentation after
you've built the branch that you're using to build the
documentation.

The documentation is built with Sphinx and has a few requirements 
(Pandoc, Numpydoc and the Nengo-Sphinx-theme).

How you install `Pandoc <http://johnmacfarlane.net/pandoc/installing.html>`_ (requirement of Sphinx)depends on your operating system,
so no instructions will be included here.
However, if you're using Windows and having a hard
time installing the requirements, please check out
`Chocolatey <https://chocolatey.org/>`_.

After you've installed Pandoc, you're ready to intall the rest of the 
requirements:

.. code-block:: bash

   pip install -r doc-requirements.txt

After you've installed all the requirements,
execute from the root directory of ``nengo-temp``
to build the documentation
which will probably take a few minutes.

.. code-block:: bash

   mkdir ./docs/build
   sphinx-build -b html ./docs ./docs/build

Now you're ready to code and look-up docs!

.. toctree::
   :maxdepth: 2

   dev_introduction
   dev_api
   nef_minimal
   simulators
   contributing
