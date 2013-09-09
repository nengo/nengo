==============================
Transitioning from the old API
==============================

The "old API" is the Python programming interface
to the Java version of Nengo;
it is described on `nengo.ca <http://nengo.ca/>`_
and the documentation hosted there.
A Theano-backed version of Nengo was created
that also implements this old API.
It is common enough that a compatibility layer,
``nengo.old_api``, was created to ensure
that old code will run until it can be update
to the new API. Not all of the old functionality
has been replicated yet.

This document describes the major differences
between the old and new APIs.

Big changes
===========

The Model abstraction
---------------------

The primary way to interact with the
new Nengo API is through the ``Model`` object.
This is analogous to the old API's
``Network`` object.

Model is a wrapper
------------------

``Model`` itself does very little;
it mostly keeps track of objects that are part of it.
This is in contrast with ``nef.Network`` that
transformed a lot of the keyword args.
Now, this is handled by the underlying Python objects.
For example::

  ens = model.make_ensemble('Ensemble', nengo.LIF(30), 1)

is functionally equivalent to::

  ens = model.add(Ensemble('Ensemble', nengo.LIF(30), 1)

Similarly for Nodes. For connections::

  conn = model.connect(ens, ens2)

is equivalent to::

  conn = ens.connect_to(ens2)

For probes::

  model.probe(ens)

is equivalent to::

  ens.probe()

The underlying implementation is a bit more complicated,
but in general, you should think of ``Model``
as a wrapper that gives shortcuts to the most commonly
used classes and functions.

No Network
----------

The old ``Network`` object has been removed entirely.
It may return, but basically we put off
implementing it for a while, until we realized
that we did not find a need for it when
creating the unit tests and examples;
likely, whatever used to be accomplished with Networks
is better implemented by making your own Nengo object
(see `Nengo objects <model_objects.html>`_),
but if that turns out to be not the case,
we'll revisit Networks.

No Origins and Terminations
---------------------------

Previously, each object had a set of origins and terminations,
which determined how the object produced output and
accepted input, respectively.
These two things have been collapsed into a single
Connection object, which contains
the logic of the origin and termination
in one place.

Because the model is defined separately
from when it's built,
the performance advantages of having
origins and terminations can be accomplished
during the build phase of the model instead.

Only Ensembles and Nodes
------------------------

Many other objects have been removed,
in order to start with a very minimal
set of objects in this first version of the API.
More objects can be added later through templates;
however, since the vast majority of models
can be defined using Ensembles and Nodes,
the API is radically simplified by only
exposing these two objects.
As we build larger models,
we can see if certain templates are used
very frequently, which may
motivate exposing them through
the Model object.

Model and Simulator separation
------------------------------

There is now a clear separation between
model definition and model creation/simulation.
The motivation behind this is to allow
for testing models as they are being created.
For example, you can create a model,
add a node and an ensemble,
and the create a simulator based
on that model and run it
to make sure that your node and ensemble
are doing what you think they're doing.
Then, you can continue adding new objects
to your model---this will not be reflected
in the simulator that you've already created,
but you can create a new simulator
with this updated model and run it
without having to rerun your script
from the top.
Basically, it allows for a more
iterative and interactive modelling process,
and makes it more explicit which
decisions are made manually and which
are automatically determined
when the simulator is created.
Additionally, this means that the
simulator timestep (dt) is not
defined until the simulator is created,
meaning that you can run the same model
with different timesteps to see if
there is a marked functional difference.

Changes to common functions
===========================

Many commonly used functions have been
simplified or changed to be more explicit.

Making ensembles
----------------

Old API signature::

  nef.Network.make(name, neurons, dimensions, tau_rc, tau_ref, max_rate, intercept, radius, encoders, decoder_noise, eval_points, noise, noise_frequency, mode, add_to_network, node_factory, decoder_sign, seed, quick, storage_code)

A simple example::

  nef.Network.make('A', 40, 1, mode='spike')

New API signature::

  model.make_ensemble(name, neurons, dimensions, **kwargs)

A simple example::

  model.make_ensemble('A', nengo.LIF(40), 1)

An important change is that the neuron model must be specified
in the ``neuron`` argument.
This is also where neuron parameters can be set. For example::

  model.make_ensemble('A', nengo.LIF(40, tau_rc=0.03), 1)

See `nonlinearities <simulator_objects.html#nonlinearities>`_
for more details.

Other properties, like the radius, encoders, etc. can still be
specified through the ``**kwargs``,
but they can also be specified after ensemble creation
through setting properties.
Every property can be set through the ``kwargs``.
For example::

  model.make_ensemble('A', nengo.LIF(40), 1, radius=1.5)

is equivalent to::

  ens = model.make_ensemble('A', nengo.LIF(40), 1)
  ens.radius = 1.5

See `Ensemble documentation <model_objects.html#ensemble>`_
for a list of properties that can be manipulated.

Making ensemble arrays (i.e., network arrays)
---------------------------------------------

Network arrays were very tightly coupled
with the old API. In the new API,
they have been decoupled and are now
an easily imported template instead.
The functionality should still be identical.

Old API::

  nef.Network.make_array(name, neurons, length, dimensions, **args)

New API::

  from nengo.templates import EnsembleArray
  model.add(EnsembleArray(name, neurons, n_ensembles, dimensions_per_ensemble, **ens_args)

See `EnsembleArray documentation <templates.html#ensemblearray>`_
for more information.

Making nodes
------------

Previously, there were several different ways
to provide input to a Nengo model:
SimpleNode, FunctionInput, and others.
All of these use cases should be covered
by the ``nengo.objects.Node``.

In the old API, you could create your own
SimpleNode, or create a FunctionInput with::

  nef.Network.make_input(name, values, zero_after_time)

In the new API, you create a node with::

  model.make_node(name, output)

where ``output`` is either a constant value
(float, list, NumPy array) or a function.

See `Node documentation <model_objects.html#node>`_
for more information.

Connecting things
-----------------

Connecting and probing things
(which are two sides of the same coin)
is now encapsulated in connection classes,
which are created by the objects
being connected to another object.
A lot of the complexity of the old API
has been pushed down to the constructors
of the connection objects. Which connection object
is created depends on that object's ``connect_to`` method.
In general, however, old API calls of the form::

  nef.Network.connect(pre, post)

will work as expected::

  model.connect(pre, post)

However, there are some changes in the additional arguments.
The old API used ``weight``, ``index_pre`` and ``index_post``
as a shortcut to define ``transform``;
in the new API, only the ``transform`` can be specified.
There are many NumPy functions that make transforms
easier to specify, but we are currently looking into
other methods of specifying the transform.

The keyword argument ``pstc`` has been renamed to ``filter``.

Aliasing
--------

We are considering moving away from string-based workflows,
as they limit what can be done with the new API.
However, for the time being, the former::

  nef.Network.set_alias(alias, node)

is now accessible as::

  model.alias(alias, node)

Under the hood changes
======================

The underlying structure of Nengo is completely different.
If you're interested, look at the
`core objects <simulator_objects.html>`_
that simulators use.
The reference simulator, ``nengo.simulator.Simulator``,
is relatively simple, save for a few methods.
It may be instructive to see
`its implementation <_modules/nengo/simulator.html>`_.
