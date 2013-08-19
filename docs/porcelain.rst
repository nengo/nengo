===============
Basic Nengo API
===============

The basic Nengo API is based on an abstraction
that we call a ``Model``. Everything that you need
for building most simple models can be done with
a ``Model``.

To create a new ``Model``::

  import nengo as ne
  model = ne.Model("My Model")

Creating Nengo objects
======================

A Nengo object is a part of your model that represents information.
There are three Nengo objects that make up a Nengo model.
Each of these objects can be constructed by calling
the appropriate ``make_`` function.

.. automethod:: nengo.api.Model.make_ensemble(name, neurons, dimensions, [max_rate, intercept, radius, encoders, neuron_model, mode])

.. automethod:: nengo.api.Model.make_node

.. automethod:: nengo.api.Model.make_network

Connecting Nengo objects
========================

Nengo objects get a lot more interesting when you connect them together.
There are two ways to connect Nengo objects together.

.. automethod:: nengo.api.Model.connect(pre, post, [function, transform, filter, learning_rule])

.. automethod:: nengo.api.Model.connect_neurons(pre, post, [weights, filter, learning_rule])

Running an experiment
=====================

Once a model has been constructed by creating Nengo objects
and connecting them together, we can run it to collect data.

First, we need to define what data we would like to collect
when we run an experiment.

.. automethod:: nengo.api.Model.probe(target, [sample_every, static=False])

Once we've probed the appropriate objects,
we can start running the experiment.

.. automethod:: nengo.api.Model.run(time, dt, [output, stop_when])

