********
Examples
********

Nengo creates these models using the principles of the
`Neural Engineering Framework <http://compneuro.uwaterloo.ca/research/nef.html>`_.
The first set of examples explains these three principles:
*representation*, *transformation*, and *dynamics*.

For a summary of these principles, see the following example:

.. toctree::

   examples/nef_summary

The following examples give a more detailed presentation:

Principle 1: Representation
===========================

.. toctree::

   examples/single_neuron
   examples/two_neurons
   examples/many_neurons
   examples/2d_representation
   examples/combining
   examples/addition

Principle 2: Transformation
===========================

.. toctree::

   examples/communication_channel
   examples/squaring
   examples/multiplication

Principle 3: Dynamics
=====================

.. toctree::

   examples/integrator
   examples/controlled_integrator
   examples/controlled_integrator2
   examples/oscillator
   examples/controlled_oscillator
   examples/lorenz_attractor

Putting these three principles together allows us to scale
these examples up to larger networks that do more complex functions.
Below are some of these complex functions,
as well as other examples that we hope will be helpful
as reference when building your own Nengo models.

Nodes
=====

.. toctree::

   examples/delay_node

Processes
=========

.. toctree::

   examples/processes

Ensembles
=========

.. toctree::

   examples/tuning_curves
   examples/izhikevich

Connections
===========

.. toctree::

   examples/inhibitory_gating

Learning
========

.. toctree::

   examples/learn_communication_channel
   examples/learn_square
   examples/learn_product
   examples/learn_unsupervised
   examples/learn_associations

Networks
========

.. toctree::

   examples/ensemble_array
   examples/matrix_multiplication
   examples/basal_ganglia
   examples/integrator_network

Semantic Pointer Architecture
=============================

.. toctree::

   examples/associative_memory
   examples/convolution
   examples/question
   examples/question_control
   examples/question_memory
   examples/spa_parser
   examples/spa_sequence
   examples/spa_sequence_routed

Extending Nengo
===============

.. toctree::

   examples/rectified_linear
