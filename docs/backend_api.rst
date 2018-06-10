*****************
Nengo backend API
*****************

.. default-role:: obj

Nengo is designed so that models created with the
:doc:`Nengo frontend API <frontend_api>`
work on a variety of different simulators, or "backends."
For example, backends have been created to take advantage of
`GPUs <https://github.com/nengo/nengo-ocl/>`_ and
`neuromorphic hardware <https://github.com/project-rig/nengo_spinnaker>`_.

Reference backend
=================

Nengo comes with a backend that is relatively fast,
and works on general purpose computers.
For most users, the only thing that you need to know
about this "reference" backend is how to
create and close a `nengo.Simulator` instance.

.. autoclass:: nengo.Simulator

.. autoclass:: nengo.simulator.ProbeDict

.. autoclass:: nengo.simulator.SignalDict

.. autoclass:: nengo.exceptions.SimulatorClosed

The build process
-----------------

The build process translates a Nengo model
to a set of data buffers (`.Signal` instances)
and computational operations (`.Operator` instances)
which implement the Nengo model
defined with the :doc:`frontend API <frontend_api>`.
The build process is central to
how the reference simulator works,
and details how Nengo can be extended to include
new neuron types, learning rules, and other components.

`Bekolay et al., 2014
<http://compneuro.uwaterloo.ca/publications/bekolay2014.html>`_
provides a high-level description
of the build process.
For lower-level details
and reference documentation, read on.

.. autoclass:: nengo.builder.Signal

.. autoclass:: nengo.builder.Operator

Operators
^^^^^^^^^

.. autoclass:: nengo.builder.operator.Reset

.. autoclass:: nengo.builder.operator.Copy

.. autoclass:: nengo.builder.operator.ElementwiseInc

.. autoclass:: nengo.builder.operator.DotInc

.. autoclass:: nengo.builder.operator.BsrDotInc

.. autoclass:: nengo.builder.operator.TimeUpdate

.. autoclass:: nengo.builder.operator.SimPyFunc

.. autoclass:: nengo.builder.neurons.SimNeurons

.. autoclass:: nengo.builder.learning_rules.SimBCM

.. autoclass:: nengo.builder.learning_rules.SimOja

.. autoclass:: nengo.builder.learning_rules.SimVoja

.. autoclass:: nengo.builder.processes.SimProcess

Build functions
^^^^^^^^^^^^^^^

.. autoclass:: nengo.builder.Builder

.. autoclass:: nengo.builder.Model

.. autofunction:: nengo.builder.network.build_network

.. autofunction:: nengo.builder.ensemble.build_ensemble

.. autoclass:: nengo.builder.ensemble.BuiltEnsemble

.. autofunction:: nengo.builder.node.build_node

.. autofunction:: nengo.builder.connection.build_connection

.. autoclass:: nengo.builder.connection.BuiltConnection

.. autofunction:: nengo.builder.probe.build_probe

.. autofunction:: nengo.builder.neurons.build_neurons

.. autofunction:: nengo.builder.neurons.build_lif

.. autofunction:: nengo.builder.neurons.build_alifrate

.. autofunction:: nengo.builder.neurons.build_alif

.. autofunction:: nengo.builder.neurons.build_izhikevich

.. autofunction:: nengo.builder.learning_rules.build_learning_rule

.. autofunction:: nengo.builder.learning_rules.build_bcm

.. autofunction:: nengo.builder.learning_rules.build_oja

.. autofunction:: nengo.builder.learning_rules.build_voja

.. autofunction:: nengo.builder.learning_rules.build_pes

.. autofunction:: nengo.builder.processes.build_process

.. autofunction:: nengo.builder.processes.build_synapse

Decoder cache
^^^^^^^^^^^^^

.. autoclass:: nengo.cache.NoDecoderCache

.. autoclass:: nengo.cache.DecoderCache

.. autoclass:: nengo.cache.Fingerprint

.. autoclass:: nengo.cache.CacheIndex

.. autoclass:: nengo.cache.WriteableCacheIndex

Optimizer
^^^^^^^^^

.. autofunction:: nengo.builder.optimizer.optimize
