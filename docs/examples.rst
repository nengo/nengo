********
Examples
********

Nengo creates these models using the principles of the
`Neural Engineering Framework <http://compneuro.uwaterloo.ca/research/nef.html>`_.
The first set of examples explains these three principles:
*representation*, *transformation*, and *dynamics*.

For a summary of these principles, see the following example:

.. toctree::
   :maxdepth: 1

   examples/advanced/nef_summary

The following examples give a more detailed presentation:

Principle 1: Representation
===========================

.. toctree::
   :maxdepth: 1

   examples/basic/single_neuron
   examples/basic/two_neurons
   examples/basic/many_neurons
   examples/basic/2d_representation
   examples/basic/combining
   examples/basic/addition

Principle 2: Transformation
===========================

.. toctree::
   :maxdepth: 1

   examples/basic/communication_channel
   examples/basic/squaring
   examples/basic/multiplication

Principle 3: Dynamics
=====================

.. toctree::
   :maxdepth: 1

   examples/dynamics/integrator
   examples/dynamics/controlled_integrator
   examples/dynamics/controlled_integrator2
   examples/dynamics/oscillator
   examples/dynamics/controlled_oscillator
   examples/dynamics/lorenz_attractor

Putting these three principles together allows us to scale
these examples up to larger networks that do more complex functions.
Below are some of these complex functions,
as well as other examples that we hope will be helpful
as reference when building your own Nengo models.

Nodes
=====

.. toctree::
   :maxdepth: 1

   examples/usage/delay_node

Processes
=========

.. toctree::
   :maxdepth: 1

   examples/advanced/processes

Ensembles
=========

.. toctree::
   :maxdepth: 1

   examples/usage/tuning_curves
   examples/advanced/izhikevich

Connections
===========

.. toctree::
   :maxdepth: 1

   examples/advanced/inhibitory_gating
   examples/advanced/functions_and_tuning_curves

Learning
========

.. toctree::
   :maxdepth: 1

   examples/learning/learn_communication_channel
   examples/learning/learn_square
   examples/learning/learn_product
   examples/learning/learn_unsupervised
   examples/learning/learn_associations

Networks
========

.. toctree::
   :maxdepth: 1

   examples/networks/ensemble_array
   examples/advanced/matrix_multiplication
   examples/networks/basal_ganglia
   examples/networks/integrator_network

Under the hood
==============

.. toctree::
   :maxdepth: 1

   examples/usage/rectified_linear
   examples/advanced/nef_algorithm

Reference
=========

.. toctree::
   :maxdepth: 1

   examples/usage/exceptions
   examples/usage/strings
