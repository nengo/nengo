=============
Nengo objects
=============

A "Nengo object" is anything in a Nengo model
that represents information.
While anything in a model can be made up of
the two objects below (``Ensemble`` and ``Node``),
more complicated objects exist that contain
many ensembles and nodes.

What makes a Nengo object a Nengo object
is that it can connect to other objects,
and be probed. In order to facilitate this,
a Nengo object should follow this template::

  class MyObject(object):
      def __init__(self, ...):
          self.connections_in = []
          self.connections_out = []
          self.probes = {}

      def connect_to(self, post, ...):
          pass

      def probe(self, to_probe, sample_every, filter):
          pass

      def build(self, model, dt):
          pass

Ensemble
========

An ensemble is a group of neurons
that collectively represent information.

.. autoclass:: nengo.objects.Ensemble
   :members:

Node
====

A node is used to provide arbitrary data to an ensemble,
or to do some other arbitrary processing.

.. autoclass:: nengo.objects.ConstantNode
   :members:

.. autoclass:: nengo.objects.Node
   :members:

===========
Connections
===========

A connection encapsulates all of the information necessary
to create a connection between two Nengo objects.
A lot of processing goes on behind the scenes
for each connection, but it is possible to
influence that processing by changing several parameters.

What makes a connection a connection is that it
keeps track of the two objects being connected,
it cannot itself be connected to something else,
and it can only be trivially probed.
In order to facilitate this,
a connection should follow this template::

  class MyConnection(object):
      def __init__(self, pre, post, ...):
          self.pre = pre
          self.post = post
          self.probes = {}

      def probe(self, to_probe, sample_every, filter):
          pass

      def build(self, model, dt):
          pass

SignalConnection
================

.. autoclass:: nengo.connections.SignalConnection
   :members:

DecodedConnection
=================

.. autoclass:: nengo.connections.DecodedConnection
   :members:

DecodedNeuronConnection
=======================

.. autoclass:: nengo.connections.DecodedNeuronConnection
   :members:

======
Probes
======

A probe is implemented as a dummy object
that can be connected to,
but can't connect to other objects.
It is implemented this way in order to leverage
the connection classes that already exist,
rather than reimplement
decoding and filtering methods.

.. autoclass:: nengo.objects.Probe
   :members:
