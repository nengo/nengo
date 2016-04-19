*****************************
Semantic Pointer Architecture
*****************************

.. default-role:: obj

The `Semantic Pointer Architecture
<http://compneuro.uwaterloo.ca/research/spa/semantic-pointer-architecture.html>`_
provides an approach to building cognitive models
implemented with large-scale spiking neural networks.

Nengo includes a ``nengo.spa`` module that provides
a simple interface for building models with
the Semantic Pointer Architecture.
See the following examples for demonstrations
of how ``nengo.spa`` works.

.. toctree::

   examples/convolution
   examples/question
   examples/question_control
   examples/question_memory
   examples/spa_sequence
   examples/spa_sequence_routed
   examples/spa_parser

API reference
=============

.. autoclass:: nengo.spa.SPA

.. autofunction:: nengo.spa.enable_spa_params

.. autoclass:: nengo.spa.SemanticPointer

.. autoclass:: nengo.spa.Vocabulary

.. autofunction:: nengo.spa.similarity

The action language
-------------------

.. autoclass:: nengo.spa.Actions

.. autoclass:: nengo.spa.actions.Action

.. autoclass:: nengo.spa.actions.Expression

.. autoclass:: nengo.spa.actions.Effect

.. autoclass:: nengo.spa.action_objects.Symbol

.. autoclass:: nengo.spa.action_objects.Source

.. autoclass:: nengo.spa.action_objects.DotProduct

.. autoclass:: nengo.spa.action_objects.Convolution

.. autoclass:: nengo.spa.action_objects.Summation

SPA modules
-----------

.. autoclass:: nengo.spa.module.Module

.. autoclass:: nengo.spa.AssociativeMemory

.. autoclass:: nengo.spa.BasalGanglia

.. autoclass:: nengo.spa.Bind

.. autoclass:: nengo.spa.Buffer

.. autoclass:: nengo.spa.Compare

.. autoclass:: nengo.spa.Cortical

.. autoclass:: nengo.spa.Input

.. autoclass:: nengo.spa.Memory

.. autoclass:: nengo.spa.State

.. autoclass:: nengo.spa.Thalamus
