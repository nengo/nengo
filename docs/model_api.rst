.. _basic-api

===============
Basic Nengo API
===============

The basic Nengo API consists
of the ``Model`` and ``Simulator`` objects;
these are the only two objects that are necessary
to learn for the majority of basic models.

``nengo.Model``
===============

Model creation
--------------

.. automethod:: nengo.Model.make_ensemble

.. automethod:: nengo.Model.make_node

.. automethod:: nengo.Model.connect

.. automethod:: nengo.Model.probe

Model manipulation
------------------

.. automethod:: nengo.Model.add

.. automethod:: nengo.Model.alias

.. automethod:: nengo.Model.get

.. automethod:: nengo.Model.get_string

.. automethod:: nengo.Model.remove


IO
--

.. automethod:: nengo.Model.save

.. automethod:: nengo.Model.load

Simulation
----------

.. automethod:: nengo.Model.prep_for_simulation

.. automethod:: nengo.Model.simulator

``nengo.simulator.Simulator``
=============================

Running simulations
-------------------

.. automethod:: nengo.simulator.Simulator.run

.. automethod:: nengo.simulator.Simulator.run_steps

.. automethod:: nengo.simulator.Simulator.step

.. automethod:: nengo.simulator.Simulator.reset

Analyzing simulations
---------------------

.. automethod:: nengo.simulator.Simulator.data

.. automethod:: nengo.simulator.Simulator.copied
