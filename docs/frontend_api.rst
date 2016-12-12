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

Distributions
=============

.. autoclass:: nengo.dists.Distribution
   :exclude-members: sample

   .. automethod:: nengo.dists.Distribution.sample(n, d=None, rng=np.random)

.. autofunction:: nengo.dists.get_samples(dist_or_samples, n, d=None, rng=np.random)

.. autoclass:: nengo.dists.Uniform

.. autoclass:: nengo.dists.Gaussian

.. autoclass:: nengo.dists.Exponential

.. autoclass:: nengo.dists.UniformHypersphere

.. autoclass:: nengo.dists.Choice

.. autoclass:: nengo.dists.Samples

.. autoclass:: nengo.dists.PDF

.. autoclass:: nengo.dists.SqrtBeta

.. autoclass:: nengo.dists.SubvectorLength

.. autoclass:: nengo.dists.CosineSimilarity

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

Processes
=========

.. autoclass:: nengo.Process

.. autoclass:: nengo.processes.PresentInput

.. autoclass:: nengo.processes.FilteredNoise

.. autoclass:: nengo.processes.BrownNoise

.. autoclass:: nengo.processes.WhiteNoise

.. autoclass:: nengo.processes.WhiteSignal

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
