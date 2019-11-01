******************
Nengo frontend API
******************

Nengo Objects
=============

.. autosummary::
   :nosignatures:

   nengo.Network
   nengo.Ensemble
   nengo.ensemble.Neurons
   nengo.Node
   nengo.Connection
   nengo.connection.LearningRule
   nengo.Probe

.. autoclass:: nengo.Network

.. autoclass:: nengo.Ensemble

.. autoclass:: nengo.ensemble.Neurons

.. autoclass:: nengo.Node

.. autoclass:: nengo.Connection

.. autoclass:: nengo.connection.LearningRule

.. autoclass:: nengo.Probe

Distributions
=============

.. automodule:: nengo.dists

   .. autoautosummary:: nengo.dists
      :nosignatures:
      :exclude-members: DistributionParam, DistOrArrayParam

Learning rule types
===================

.. automodule:: nengo.learning_rules

   .. autoautosummary:: nengo.learning_rules
      :nosignatures:
      :exclude-members: LearningRuleTypeParam, LearningRuleTypeSizeInParam

Neuron types
============

.. Note: we don't use automodule/autoautosummary here because we want the
   canonical reference to these objects to be ``nengo.Class`` even though
   they actually live in ``nengo.neurons.Class``.

.. autosummary::
   :nosignatures:

   nengo.neurons.NeuronType
   nengo.neurons.settled_firingrate
   nengo.Direct
   nengo.RectifiedLinear
   nengo.SpikingRectifiedLinear
   nengo.Sigmoid
   nengo.LIF
   nengo.LIFRate
   nengo.AdaptiveLIF
   nengo.AdaptiveLIFRate
   nengo.Izhikevich

.. autoclass:: nengo.neurons.NeuronType

.. autofunction:: nengo.neurons.settled_firingrate

.. autoclass:: nengo.Direct

.. autoclass:: nengo.RectifiedLinear

.. autoclass:: nengo.SpikingRectifiedLinear

.. autoclass:: nengo.Sigmoid

.. autoclass:: nengo.LIF

.. autoclass:: nengo.LIFRate

.. autoclass:: nengo.AdaptiveLIF

.. autoclass:: nengo.AdaptiveLIFRate

.. autoclass:: nengo.Izhikevich

.. autoclass:: nengo.neurons.NeuronTypeParam

Processes
=========

.. automodule:: nengo.processes

   .. autoautosummary:: nengo.processes
      :nosignatures:
      :exclude-members: PiecewiseDataParam

      nengo.Process

.. autoclass:: nengo.Process

Solvers
=======

.. automodule:: nengo.solvers

   .. autoautosummary:: nengo.solvers
      :nosignatures:
      :exclude-members: SolverParam

Solver methods
--------------

.. automodule:: nengo.utils.least_squares_solvers

   .. autoautosummary:: nengo.utils.least_squares_solvers
      :nosignatures:

Synapse models
==============

.. Note: we don't use automodule/autoautosummary here because we want the
   canonical reference to these objects to be ``nengo.Class`` even though
   they actually live in ``nengo.synapses.Class``.

.. autosummary::
   :nosignatures:

   nengo.synapses.Synapse
   nengo.LinearFilter
   nengo.Lowpass
   nengo.Alpha
   nengo.synapses.Triangle

.. autoclass:: nengo.synapses.Synapse

.. autoclass:: nengo.LinearFilter

.. autoclass:: nengo.Lowpass

.. autoclass:: nengo.Alpha

.. autoclass:: nengo.synapses.Triangle

.. autoclass:: nengo.synapses.SynapseParam

Transforms
==========

.. Note: we don't use automodule/autoautosummary here because we want the
   canonical reference to these objects to be ``nengo.Class`` even though
   they actually live in ``nengo.synapses.Class``.

.. autosummary::
   :nosignatures:

   nengo.transforms.Transform
   nengo.Dense
   nengo.Sparse
   nengo.transforms.SparseMatrix
   nengo.Convolution
   nengo.transforms.ChannelShape

.. autoclass:: nengo.transforms.Transform

.. autoclass:: nengo.Dense

.. autoclass:: nengo.Sparse

.. autoclass:: nengo.transforms.SparseMatrix

.. autoclass:: nengo.transforms.SparseInitParam

.. autoclass:: nengo.Convolution

.. autoclass:: nengo.transforms.ChannelShape

.. autoclass:: nengo.transforms.ChannelShapeParam
