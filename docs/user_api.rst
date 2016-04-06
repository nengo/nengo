*******************
Nengo Modelling API
*******************

Objects
=======

Network
-------

.. autoclass:: nengo.network.Network
   :members:

Ensemble
--------

An ensemble is a group of neurons
that collectively represent information.

.. autoclass:: nengo.ensemble.Ensemble
   :members:

Node
----

A node is used to provide arbitrary data to an ensemble,
or to do some other arbitrary processing.

.. autoclass:: nengo.node.Node
   :members:

Connection
----------

A connection encapsulates all of the information necessary
to create a connection between two Nengo objects.
A lot of processing goes on behind the scenes
for each connection, but it is possible to
influence that processing by changing several parameters.

.. autoclass:: nengo.connection.Connection
   :members:

Probe
-----

.. autoclass:: nengo.probe.Probe
   :members:

Neuron types
============

.. autoclass:: nengo.neurons.NeuronType
   :members:

.. autoclass:: nengo.neurons.Direct
   :members:

.. autoclass:: nengo.neurons.RectifiedLinear
   :members:

.. autoclass:: nengo.neurons.Sigmoid
   :members:

.. autoclass:: nengo.neurons.LIF
   :members:

.. autoclass:: nengo.neurons.LIFRate
   :members:

.. autoclass:: nengo.neurons.AdaptiveLIF
   :members:

.. autoclass:: nengo.neurons.AdaptiveLIFRate
   :members:

.. autoclass:: nengo.neurons.Izhikevich
   :members:

Learning rule types
===================

.. autoclass:: nengo.learning_rules.LearningRuleType

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

Decoder solvers
===============

.. autoclass:: nengo.solvers.Lstsq
   :members:

.. autoclass:: nengo.solvers.LstsqNoise
   :members:

.. autoclass:: nengo.solvers.LstsqMultNoise
   :members:

.. autoclass:: nengo.solvers.LstsqL2
   :members:

.. autoclass:: nengo.solvers.LstsqL2nz
   :members:

.. autoclass:: nengo.solvers.LstsqL1
   :members:

.. autoclass:: nengo.solvers.LstsqDrop
   :members:

.. autoclass:: nengo.solvers.Nnls
   :members:

.. autoclass:: nengo.solvers.NnlsL2
   :members:

.. autoclass:: nengo.solvers.NnlsL2nz
   :members:

Simulator
=========

.. autoclass:: nengo.simulator.Simulator
   :members:
