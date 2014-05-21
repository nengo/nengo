*******************
Nengo Modelling API
*******************

TODO intro

Objects
=======

TODO intro

Network
-------

.. autoclass:: nengo.objects.Network
   :members:

Ensemble
--------

An ensemble is a group of neurons
that collectively represent information.

.. autoclass:: nengo.objects.Ensemble
   :members:

Node
----

A node is used to provide arbitrary data to an ensemble,
or to do some other arbitrary processing.

.. autoclass:: nengo.objects.Node
   :members:

Connection
----------

A connection encapsulates all of the information necessary
to create a connection between two Nengo objects.
A lot of processing goes on behind the scenes
for each connection, but it is possible to
influence that processing by changing several parameters.

.. autoclass:: nengo.objects.Connection
   :members:

Probe
-----

.. autoclass:: nengo.objects.Probe
   :members:

Neurons
=======

.. autoclass:: nengo.neurons.Direct
   :members:

.. autoclass:: nengo.neurons.LIF
   :members:

.. autoclass:: nengo.neurons.LIFRate
   :members:

.. autoclass:: nengo.neurons.AdaptiveLIF
   :members:

.. autoclass:: nengo.neurons.AdaptiveLIFRate
   :members:

Learning rules
==============

.. autoclass:: nengo.learning_rules.PES
   :members:

.. autoclass:: nengo.learning_rules.BCM
   :members:

.. autoclass:: nengo.learning_rules.Oja
   :members:

Synapses
========

.. autoclass:: nengo.synapses.Lowpass
   :members:

.. autoclass:: nengo.synapses.Alpha
   :members:

.. autoclass:: nengo.synapses.LinearFilter
   :members:

Solvers
=======

.. autoclass:: nengo.decoders.Lstsq
   :members:

.. autoclass:: nengo.decoders.LstsqNoise
   :members:

.. autoclass:: nengo.decoders.LstsqMultNoise
   :members:

.. autoclass:: nengo.decoders.LstsqL2
   :members:

.. autoclass:: nengo.decoders.LstsqL2nz
   :members:

.. autoclass:: nengo.decoders.LstsqL1
   :members:

.. autoclass:: nengo.decoders.LstsqDrop
   :members:

.. autoclass:: nengo.decoders.Nnls
   :members:

.. autoclass:: nengo.decoders.NnlsL2
   :members:

.. autoclass:: nengo.decoders.NnlsL2nz
   :members:

Simulator
=========

.. autoclass:: nengo.simulator.Simulator
   :members:
