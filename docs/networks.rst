********
Networks
********

Networks are an abstraction of a grouping of nengo objects
(Nodes, Ensembles, Connections).
Like all abstractions, this helps with code-reuse and maintainability.
You'll find the documentation for the various pre-built networks below.

Building your own network can be a great way to encapsulate
parts of your model, making your code easier to understand,
easier to re-use, and easier to share.
The following examples will help you to build your own networks:

.. toctree::

   examples/network_design
   examples/network_design_advanced

You may also find the following documentation on the ``config`` system
useful:

.. toctree::

   examples/config


Ensemble Array
==============

.. autoclass:: nengo.networks.EnsembleArray
   :members:

Action Selection
================

.. autofunction:: nengo.networks.BasalGanglia

.. autofunction:: nengo.networks.Thalamus

Working Memory
==============

.. autofunction:: nengo.networks.InputGatedMemory

Product
=======

.. autofunction:: nengo.networks.Product

Circular Convolution
====================

.. autofunction:: nengo.networks.CircularConvolution

Integrator
==========

.. autofunction:: nengo.networks.Integrator

Oscillator
==========

.. autofunction:: nengo.networks.Oscillator
