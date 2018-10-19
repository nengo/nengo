******************
Nengo frontend API
******************

.. default-role:: obj

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

.. autosummary::
   :nosignatures:

   nengo.dists.Distribution
   nengo.dists.get_samples
   nengo.dists.Uniform
   nengo.dists.Gaussian
   nengo.dists.Exponential
   nengo.dists.UniformHypersphere
   nengo.dists.Choice
   nengo.dists.Samples
   nengo.dists.PDF
   nengo.dists.SqrtBeta
   nengo.dists.SubvectorLength
   nengo.dists.CosineSimilarity

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

Transforms
==========

.. autosummary::
   :nosignatures:

   nengo.transforms.Transform
   nengo.transforms.get_transform
   nengo.transforms.Convolution
   nengo.transforms.ConvShape

.. autoclass:: nengo.transforms.Transform
   :exclude-members: sample

   .. automethod:: nengo.transforms.Transform.sample(rng=np.random)

.. autofunction:: nengo.transforms.get_transform(transform, shape, rng=np.random)

.. autoclass:: nengo.transforms.Convolution

.. autoclass:: nengo.transforms.ConvShape

Neuron types
============

.. autosummary::
   :nosignatures:

   nengo.neurons.NeuronType
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

.. autoclass:: nengo.Direct

.. autoclass:: nengo.RectifiedLinear

.. autoclass:: nengo.SpikingRectifiedLinear

.. autoclass:: nengo.Sigmoid

.. autoclass:: nengo.LIF

.. autoclass:: nengo.LIFRate

.. autoclass:: nengo.AdaptiveLIF

.. autoclass:: nengo.AdaptiveLIFRate

.. autoclass:: nengo.Izhikevich

Learning rule types
===================

.. autosummary::
   :nosignatures:

   nengo.learning_rules.LearningRuleType
   nengo.PES
   nengo.BCM
   nengo.Oja
   nengo.Voja

.. autoclass:: nengo.learning_rules.LearningRuleType

.. autoclass:: nengo.PES

.. autoclass:: nengo.BCM

.. autoclass:: nengo.Oja

.. autoclass:: nengo.Voja

Processes
=========

.. autosummary::
   :nosignatures:

   nengo.Process
   nengo.processes.PresentInput
   nengo.processes.FilteredNoise
   nengo.processes.BrownNoise
   nengo.processes.WhiteNoise
   nengo.processes.WhiteSignal
   nengo.processes.Piecewise

.. autoclass:: nengo.Process

.. autoclass:: nengo.processes.PresentInput

.. autoclass:: nengo.processes.FilteredNoise

.. autoclass:: nengo.processes.BrownNoise

.. autoclass:: nengo.processes.WhiteNoise

.. autoclass:: nengo.processes.WhiteSignal

.. autoclass:: nengo.processes.Piecewise

Synapse models
==============

.. autosummary::
   :nosignatures:

   nengo.synapses.Synapse
   nengo.synapses.filt
   nengo.synapses.filtfilt
   nengo.LinearFilter
   nengo.Lowpass
   nengo.Alpha
   nengo.synapses.Triangle

.. autoclass:: nengo.synapses.Synapse

.. autofunction:: nengo.synapses.filt

.. autofunction:: nengo.synapses.filtfilt

.. autoclass:: nengo.LinearFilter

.. autoclass:: nengo.Lowpass

.. autoclass:: nengo.Alpha

.. autoclass:: nengo.synapses.Triangle

Decoder and connection weight solvers
=====================================

.. autosummary::
   :nosignatures:

   nengo.solvers.Solver
   nengo.solvers.Lstsq
   nengo.solvers.LstsqNoise
   nengo.solvers.LstsqMultNoise
   nengo.solvers.LstsqL2
   nengo.solvers.LstsqL2nz
   nengo.solvers.LstsqL1
   nengo.solvers.LstsqDrop
   nengo.solvers.Nnls
   nengo.solvers.NnlsL2
   nengo.solvers.NnlsL2nz
   nengo.solvers.NoSolver
   nengo.utils.least_squares_solvers.LeastSquaresSolver
   nengo.utils.least_squares_solvers.Cholesky
   nengo.utils.least_squares_solvers.ConjgradScipy
   nengo.utils.least_squares_solvers.LSMRScipy
   nengo.utils.least_squares_solvers.Conjgrad
   nengo.utils.least_squares_solvers.BlockConjgrad
   nengo.utils.least_squares_solvers.SVD
   nengo.utils.least_squares_solvers.RandomizedSVD


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

.. autoclass:: nengo.solvers.NoSolver

.. autoclass:: nengo.utils.least_squares_solvers.LeastSquaresSolver

.. autoclass:: nengo.utils.least_squares_solvers.Cholesky

.. autoclass:: nengo.utils.least_squares_solvers.ConjgradScipy

.. autoclass:: nengo.utils.least_squares_solvers.LSMRScipy

.. autoclass:: nengo.utils.least_squares_solvers.Conjgrad

.. autoclass:: nengo.utils.least_squares_solvers.BlockConjgrad

.. autoclass:: nengo.utils.least_squares_solvers.SVD

.. autoclass:: nengo.utils.least_squares_solvers.RandomizedSVD
