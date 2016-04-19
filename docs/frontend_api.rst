*******************
Nengo Modelling API
*******************

.. default-role:: obj

Nengo Objects
=============

.. autoclass:: nengo.Network

.. autoclass:: nengo.Ensemble

.. autoclass:: nengo.ensemble.Neurons

.. autoclass:: nengo.Node

.. autoclass:: nengo.Connection

.. autoclass:: nengo.connection.LearningRule

.. autoclass:: nengo.Probe

Neuron types
============

.. autoclass:: nengo.neurons.NeuronType

.. autoclass:: nengo.Direct

.. autoclass:: nengo.RectifiedLinear

.. autoclass:: nengo.Sigmoid

.. autoclass:: nengo.LIF

.. autoclass:: nengo.LIFRate

.. autoclass:: nengo.AdaptiveLIF

.. autoclass:: nengo.AdaptiveLIFRate

.. autoclass:: nengo.Izhikevich

Learning rule types
===================

.. autoclass:: nengo.learning_rules.LearningRuleType

.. autoclass:: nengo.PES

.. autoclass:: nengo.BCM

.. autoclass:: nengo.Oja

.. autoclass:: nengo.Voja

Synapse models
==============

.. autoclass:: nengo.synapses.Synapse

.. autofunction:: nengo.synapses.filt

.. autofunction:: nengo.synapses.filtfilt

.. autoclass:: nengo.LinearFilter

.. autoclass:: nengo.Lowpass

.. autoclass:: nengo.Alpha

.. autoclass:: nengo.synapses.Triangle

Decoder and connection weight solvers
=====================================

.. autoclass:: nengo.solvers.Solver
   :special-members: __call__

.. autoclass:: nengo.solvers.Lstsq

.. autoclass:: nengo.solvers.LstsqNoise

.. autoclass:: nengo.solvers.LstsqMultNoise

.. autoclass:: nengo.solvers.LstsqL2

.. autoclass:: nengo.solvers.LstsqL2nz

.. autoclass:: nengo.solvers.LstsqL1

.. autoclass:: nengo.solvers.LstsqDrop

.. autoclass:: nengo.solvers.Nnls

.. autoclass:: nengo.solvers.NnlsL2

.. autoclass:: nengo.solvers.NnlsL2nz
