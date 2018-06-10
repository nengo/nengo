*******************************
Setting parameters with Configs
*******************************

.. default-role:: obj

Building models with the :doc:`Nengo frontend API <frontend_api>`
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

   examples/usage/config

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

Quirks
======

.. toctree::

   examples/quirks/config

Parameters
==========

Under the hood, Nengo objects store information
using `.Parameter` instances,
which are also used by the config system.
Most users will not need to know about
`.Parameter` objects.

.. autoclass:: nengo.params.Parameter

.. autoclass:: nengo.params.ObsoleteParam

.. autoclass:: nengo.params.BoolParam

.. autoclass:: nengo.params.NumberParam

.. autoclass:: nengo.params.IntParam

.. autoclass:: nengo.params.StringParam

.. autoclass:: nengo.params.EnumParam

.. autoclass:: nengo.params.TupleParam

.. autoclass:: nengo.params.ShapeParam

.. autoclass:: nengo.params.DictParam

.. autoclass:: nengo.params.NdarrayParam

.. autoclass:: nengo.params.FunctionParam

.. autoclass:: nengo.exceptions.ValidationError
