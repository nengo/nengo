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

   examples/advanced/nef-summary

The following examples give a more detailed presentation:

Principle 1: Representation
===========================

.. toctree::
   :maxdepth: 1

   examples/basic/single-neuron
   examples/basic/two-neurons
   examples/basic/many-neurons
   examples/basic/2d-representation
   examples/basic/combining
   examples/basic/addition

Principle 2: Transformation
===========================

.. toctree::
   :maxdepth: 1

   examples/basic/communication-channel
   examples/basic/squaring
   examples/basic/multiplication

Principle 3: Dynamics
=====================

.. toctree::
   :maxdepth: 1

   examples/dynamics/integrator
   examples/dynamics/controlled-integrator
   examples/dynamics/controlled-integrator2
   examples/dynamics/oscillator
   examples/dynamics/controlled-oscillator
   examples/dynamics/lorenz-attractor

Putting these three principles together allows us to scale
these examples up to larger networks that do more complex functions.
Below are some of these complex functions,
as well as other examples that we hope will be helpful
as reference when building your own Nengo models.

Nodes
=====

.. toctree::
   :maxdepth: 1

   examples/usage/delay-node

Processes
=========

.. toctree::
   :maxdepth: 1

   examples/advanced/processes

Ensembles
=========

.. toctree::
   :maxdepth: 1

   examples/usage/tuning-curves
   examples/advanced/izhikevich

Connections
===========

.. toctree::
   :maxdepth: 1

   examples/advanced/inhibitory-gating
   examples/advanced/functions-and-tuning-curves

Learning
========

.. toctree::
   :maxdepth: 1

   examples/learning/learn-communication-channel
   examples/learning/learn-square
   examples/learning/learn-product
   examples/learning/learn-unsupervised
   examples/learning/learn-associations

Networks
========

.. toctree::
   :maxdepth: 1

   examples/networks/ensemble-array
   examples/advanced/matrix-multiplication
   examples/networks/basal-ganglia
   examples/networks/integrator-network

Under the hood
==============

.. toctree::
   :maxdepth: 1

   examples/usage/rectified-linear
   examples/advanced/nef-algorithm

Reference
=========

.. toctree::
   :maxdepth: 1

   examples/usage/exceptions
   examples/usage/strings
