***************************
Introduction for developers
***************************

Let's start off with some basics
in case you missed them in the README.

Developer installation
======================

If you want to change parts of Nengo,
you should do a developer installation.

.. code-block:: bash

   git clone https://github.com/nengo/nengo.git
   cd nengo
   python setup.py develop --user

If you use a ``virtualenv`` (recommended!)
you can omit the ``--user`` flag.

How to build the documentation
==============================

The documentation is built with Sphinx and has a few requirements
(Pandoc, Numpydoc and the sphinx_rtd_theme).

How you install `Pandoc <http://johnmacfarlane.net/pandoc/installing.html>`_
(requirement of ``IPython.nbconvert``) depends on your operating system,
and personal preference. Briefly, we recommend checking out the following
tools for installing Pandoc:

- Linux: your distribution's package manager (``apt-get``, ``yum``, etc)
- Mac OS X: `Homebrew <http://brew.sh/>`_
- Windows: `Chocolatey <https://chocolatey.org/>`_.

After you've installed Pandoc, you're ready to intall the rest of the
requirements:

.. code-block:: bash

   pip install -r requirements-docs.txt

After you've installed all the requirements,
run the following command from the root directory of ``nengo``
to build the documentation.
It will take a few minutes, as all examples are run
as part of the documentation building process.

.. code-block:: bash

   python setup.py build_sphinx

Now you're ready to code and look-up docs!
