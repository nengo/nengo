*******************************
Setting parameters with Configs
*******************************

.. default-role:: obj

Building models with the :doc:`modelling API <frontend_api>`
involves constructing many objects,
each with many parameters that can be set.
To make setting all of these parameters easier,
Nengo has a ``config`` system and
pre-set configurations.

The ``config`` system
=====================

Nengo's ``config`` system is used for two important functions:

1. Setting default parameters with a hierarchy of defaults.
2. Associating new information with Nengo classes and objects
   without modifying the classes and objects themselves.

A tutorial-style introduction to the ``config`` system
can be found below:

.. toctree::

   examples/config

``config`` system API
---------------------

.. autoclass:: nengo.Config

.. autoclass:: nengo.config.ClassParams

.. autoclass:: nengo.config.InstanceParams

Preset configs
==============

Nengo includes preset configurations that can be
dropped into your model to enable specific neural circuits.

.. autofunction:: nengo.presets.ThresholdingEnsembles
