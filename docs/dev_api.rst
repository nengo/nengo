************
Core objects
************

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.

TODO more

TODO block diagram?

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

.. autofunction:: nengo.builder.build_filter_synapse

.. autofunction:: nengo.builder.build_lowpass_synapse

.. autofunction:: nengo.builder.build_alpha_synapse


Signals
=======

.. autoclass:: nengo.builder.Signal
   :members:

.. autoclass:: nengo.builder.SignalView
   :members:

Operators
=========

.. autoclass:: nengo.builder.Operator
   :members:

.. autoclass:: nengo.builder.Reset
   :members:

.. autoclass:: nengo.builder.Copy
   :members:

.. autoclass:: nengo.builder.DotInc
   :members:

.. autoclass:: nengo.builder.ProdUpdate
   :members:

.. autoclass:: nengo.builder.SimPyFunc
   :members:

.. autoclass:: nengo.builder.SimNeurons
   :members:

.. autoclass:: nengo.builder.SimBCM
   :members:

.. autoclass:: nengo.builder.SimOja
   :members:

.. autoclass:: nengo.builder.SimFilterSynapse
   :members:
