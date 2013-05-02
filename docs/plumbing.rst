==================
Advanced Nengo API
==================

In the advanced Nengo API, we make available some additional
properties in all of the Nengo objects.
In some cases, you can extend the behaviour of these objects
by making subclasses of them.

Model
=====

In addition to the methods described in the `basic API <porcelain.html>`_,
``Model`` contains a few additional methods to facilitate
creating more complicated Nengo objects and incorporating them
with the objects created with ``Model.make_`` calls.

.. automethod:: nengo.api.Model.add

.. automethod:: nengo.api.Model.get

.. automethod:: nengo.api.Model.remove

Once your model gets large and complex, it can be difficult
to reference ensembles deeply nested in networks within networks.
To alleviate this, you can create aliases to Nengo objects.

.. automethod:: nengo.api.Model.make_alias

Network
=======

The model creation functions of ``Model`` are wrapper
functions that call the same functions in the ``Network``.
In other words, a ``Model`` is just a special network
that can be built and run.
We include these (hopefully now familiar) methods
below for completeness.

.. autoclass:: nengo.api.Network
   :members:

Ensemble
========

An ensemble is a group of neurons
that collectively represent information.
TODO: more

.. autoclass:: nengo.api.Ensemble
   :members:

Node
====

A node is used to provide arbitrary data to an ensemble,
or to do some other arbitrary processing.
TODO: more

.. autoclass:: nengo.api.Node
   :members:

Connection
==========

A connection encapsulates all of the information necessary
to create a connection between two Nengo objects.
A lot of processing goes on behind the scenes
for each connection, but it is possible to
influence that processing by changing several parameters.


.. autoclass:: nengo.api.Connection
   :members:

