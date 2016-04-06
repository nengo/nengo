***************
Getting Started
***************

Installation
============

Using the basic functions of Nengo requires ``NumPy``.
To do any kind of visualization, you will need ``matplotlib``.
It is highly recommended that ``IPython`` is installed as
well, in order to fully appreciate the IPython notebook
examples. For Python beginners, an all-in-one solution
like `Anaconda <https://store.continuum.io/cshop/anaconda/>`_
is recommended to install these packages, as well as
Python itself.

Basic installation
------------------

To install Nengo, we recommend using ``pip``.

.. code:: bash

   pip install nengo

Developer installation
----------------------

If you plan to make changes to Nengo,
you should clone its git repository
and do a developer (editable) install instead.

.. code:: bash

   git clone https://github.com/nengo/nengo.git
   cd nengo
   python setup.py develop --user

If you're using a ``virtualenv``
(recommended!) then you can omit the ``--user`` flag.

Usage
=====

Everything in a Nengo model is contained within a
:class:`nengo.Network`. To create a new ``Network``::

  import nengo
  model = nengo.Network()

Creating Nengo objects
----------------------

A Nengo object is a part of your model that represents information.
When creating a new object, you must place it within a ``with``
block in order to inform Nengo which network your object
should be placed in.

There are two objects that make up a basic Nengo model.
A :class:`nengo.Ensemble` is a group of neurons that represents
information in the form of real valued numbers.

::

  with model:
      my_ensemble = nengo.Ensemble(n_neurons=40, dimensions=1)

In this case, ``my_ensemble`` is made up of
40 neurons (by default, Nengo uses leaky integrate-and-fire neurons)
and it is representing a one dimensional signal.
In other words, this ensemble represents a single number.

In order to provide input to this ensemble
(to emulate some signal that exists in nature, for example)
we create a :class:`Node`.

::

  with model:
      my_node = nengo.Node(output=0.5)

In this case, ``my_node`` emits the number 0.5.

In most cases, however, we want more dynamic information.
We can make a :class:`nengo.Node` using a function as output
instead of a number.

::

  import numpy as np
  with model:
      sin_node = nengo.Node(output=np.sin)

This node will represent a sine wave.

Connecting Nengo objects
------------------------

We can connect nodes to ensembles
in order to represent that information
in the activity a group of neurons.

::

  with model:
      nengo.Connection(my_node, my_ensemble)

This connects ``my_node`` to ``my_ensemble``,
meaning that ``my_ensemble`` will now represent
0.5 in its population of 40 neurons.

Ensembles can also be connected to other models.
When the dimensionality of the objects being
connectd are different, we can use Python's
slice syntax to route information from
one node or ensemble to another.
For example::

  with model:
      two_d_ensemble = nengo.Ensemble(n_neurons=80, dimensions=2)
      nengo.Connection(sin_node, two_d_ensemble[0])
      nengo.Connection(my_ensemble, two_d_ensemble[1])

This creates a new ensemble that represents
two real-valued signals.
By connecting ``sin_node`` to ``two_d_ensemble``,
its first dimension now represents a sine wave.
Its second dimensions now represents the same
value as ``my_ensemble``.

When creating connections,
we can specify a function that
will be computed across the connection.


::

  with model:
      square = nengo.Ensemble(n_neurons=40, dimensions=1)
      nengo.Connection(my_ensemble, square, function=np.square)

Functions can be computed over multiple dimensions, as well.

::

  def product(x):
      return x[0] * x[1]

  with model:
      product_ensemble = nengo.Ensemble(n_neurons=40, dimensions=1)
      nengo.Connection(two_d_ensemble, product_ensemble, function=product)

Probing Nengo objects
---------------------

Once you have defined the objects in your model
and how they're connected,
you can decide what data you want to collect
by probing those objects.

If we wanted to collect data from
our 2D Ensemble and the Product of those two dimensions::

  with model:
      two_d_probe = nengo.Probe(two_d_ensemble, synapse=0.01)
      product_probe = nengo.Probe(product_ensemble, synapse=0.01)

The argument ``synapse`` defines the time constant
on a causal low-pass filter,
which approximates a simple synapse model.
The output of ensembles of spiking neurons
can be very noisy, so a filter is recommended.

Running an experiment
---------------------

Once a model has been constructed and we have probed
certain objects, we can run it to collect data.

To run a model, we must first build a simulator
based on the model we've defined.

::

  sim = nengo.Simulator(model)

We can then run that simulator.
For example, to run our model for five seconds::

  sim.run(5.0)

Once a simulation has been run at least once
(it can be run for additional time if desired)
the data collected can be accessed
for analysis or visualization.

::

  print(sim.data[product_probe][-10:])

For more details on these objects,
see `the API documentation <user_api.html>`_.

Next steps
==========

* If you're wondering how this works and you're not
  familiar with the Neural Engineering Framework,
  we recommend reading
  `this technical overview <http://compneuro.uwaterloo.ca/files/publications/stewart.2012d.pdf>`_.
* If you have some understanding of the NEF already,
  or just want to dive in headfirst,
  check out `our extensive set of examples <examples.html>`_.
* If you want to see the real capabilities of Nengo, see our
  `publications created with the NEF and Nengo <http://compneuro.uwaterloo.ca/publications.html>`_.
