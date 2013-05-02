The Nengo API
==================================

Nengo is a suite of software used to build and simulate
large-scale brain models using the methods of the
`Neural Engineering Framework
<http://ctnsrv.uwaterloo.ca/cnrglab/node/215>`_.

If you're new to Nengo, start with the basic API
and the associated examples.

.. toctree::
   :maxdepth: 2

   porcelain
   string_reference
   basic_examples

Once you're comfortable with making simple Nengo models,
you can start making more advanced simulations by
setting additional options on the Nengo objects
in your model.
   
.. toctree::
   :maxdepth: 2

   plumbing
   advanced_examples

If the advanced API is not sufficient for the model
that you want to create, then you have to look past
the API to how specific backends implement the Nengo API.
The functionality that you want may already exist in
one of the backends, but is not common enough to
include in the Nengo API.

.. toctree::
   :maxdepth: 2

   python_backend
   jython_backend
   theano_backend

We welcome contributions to the Nengo API
or to any of the backends! Please read the documents below
to get acquainted with how we develop Nengo,
and to see how you can help.

.. toctree::
   :maxdepth: 2

   history
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

