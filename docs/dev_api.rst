************
Core objects
************

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.

Model
=====

.. autoclass:: nengo.builder.Model
   :members:

Builder
=======

.. autoclass:: nengo.builder.Builder
   :members:

Build functions
---------------

Nengo Objects
^^^^^^^^^^^^^

.. autofunction:: nengo.builder.build_network

.. autofunction:: nengo.builder.build_ensemble

.. autofunction:: nengo.builder.build_node

.. autofunction:: nengo.builder.build_probe

.. autofunction:: nengo.builder.build_connection

Neurons
^^^^^^^

.. autofunction:: nengo.builder.build_lifrate

.. autofunction:: nengo.builder.build_lif

.. autofunction:: nengo.builder.build_alifrate

.. autofunction:: nengo.builder.build_alif


Learning rules
^^^^^^^^^^^^^^

.. autofunction:: nengo.builder.build_pes

.. autofunction:: nengo.builder.build_bcm

.. autofunction:: nengo.builder.build_oja


Synapses
^^^^^^^^

.. autofunction:: nengo.builder.build_synapse


Signals
=======

.. autoclass:: nengo.builder.signal.Signal
   :members:

.. autoclass:: nengo.builder.signal.SignalView
   :members:

Operators
=========

.. autoclass:: nengo.builder.operator.Operator
   :members:

.. autoclass:: nengo.builder.operator.Reset
   :members:

.. autoclass:: nengo.builder.operator.Copy
   :members:

.. autoclass:: nengo.builder.operator.DotInc
   :members:

.. autoclass:: nengo.builder.node.SimPyFunc
   :members:

.. autoclass:: nengo.builder.neurons.SimNeurons
   :members:

.. autoclass:: nengo.builder.learning_rules.SimOja
   :members:

.. autoclass:: nengo.builder.synapses.SimSynapse
   :members:
