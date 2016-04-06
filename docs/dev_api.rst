***********************
Reference simulator API
***********************

Understanding how the reference simulator works
is important for debugging problems,
and implementing your own simulator.

In general, there are two steps to the reference simulator.
The first is a build step, in which a ``Network``
is converted into a ``Model`` which consists of
``Signals`` (values that can be manipulated)
and ``Operators`` (operations to be done on those values).
The second is the simulator, which runs
``Operator`` functions and collects probed data.
The simulator API is described in the
`user API <user_api.html>`_.

`Bekolay et al., 2014 <http://compneuro.uwaterloo.ca/publications/bekolay2014.html>`_
provides a high-level description
and detailed picture of the build process,
which may helpful.

Build step
==========

.. autoclass:: nengo.builder.Model
   :members:

.. autoclass:: nengo.builder.Builder
   :members:

Signals
-------

.. autoclass:: nengo.builder.signal.Signal
   :members:

.. autoclass:: nengo.builder.signal.SignalView
   :members:

Operators
---------

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

.. autoclass:: nengo.builder.learning_rules.SimBCM
   :members:

.. autoclass:: nengo.builder.synapses.SimSynapse
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
