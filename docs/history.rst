*************
Nengo history
*************

Some form of Nengo has existed since 2003.
From then until now (April, 2013 as of the creation of this document)
students and researchers have used Nengo to create models
which help us understand the brain.
Nengo has evolved with that understanding.

We're currently in what could be called the third generation
of Nengo development. Many of the design decisions of this generation
are a result of lessons learnt from the first two generations.

Summary:

Generation 1: NESim (Matlab) -> Nemo (Matlab)

Generation 2: NEO (Java) -> Nengo (Java) -> Nengo GUI (Piccolo2D)

Generation 3: Nengo scripting layer (Jython) -> Nengo Theano backend (Python)

Generation 4: Nengo API -> Nengo reference implementation (Python)
                        -> Nengo reference implementation (Jython)
                        -> Nengo backends (Theano, PyNN)

Generation 5: Nengo 2.0

Generation 1
============

When Chris Eliasmith and Charles H. Anderson released the book
??cite?? Neural Engineering: .. (2003),
they described a framework for creating models of spiking neurons
called the Neural Engineering Framework.
With the book, they provided a set of Matlab scripts
to facilitate the use of these methods in theoretical neuroscience research.

NESim (Neural Engineering Simulator) was that set of Matlab scripts,
along with a basic graphical user interface that allowed users
to set the parameters of simulations.
Its primary developers were Chris Eliasmith and Bryan Tripp (???).

???screenshot???

NESim offered fast simulations by leveraging the computational power
of Matlab. And while the GUI may not have been pretty,
it enabled researchers to quickly test out ideas before
making a full model. The use of Matlab also meant that researchers could
do their simulation and analysis in the same environment.

Nemo stuff

However, there were some downsides to NESim. It was difficult
to communicate with other simulation environments.
The GUI, while functional, was not very eye-catching or dynamic.
The object model of Matlab is limited.
And, even though NESim is open source software,
Matlab itself is not, meaning that NESim
was not accessible to everyone.

Important publications that used NESim:

* cite1
* cite2

Generation 2
============

The desire for a more robust object hierarchy
and to interact with other simulation environments
resulted in the creation of NEO (Neural Engineering Objects)
in 2007 by Bryan Tripp. NEO was later renamed to Nengo,
also a short form of Neural Engineering Objects.

Nengo was created in Java, which encourages the use
of deeply nested object hierarchies.
This hierarchy of objects allowed Nengo to be flexible;
it could implement the same types of simulations as
NESim did, but it could also have those simulated objects
communicate with other types of objects.
Also, despite a common misconception about the speed of Java,
simulations in Nengo maintained the speed of the Matlab NESim.
Java, like Matlab, is multi-platform, meaning it would work
on any modern operating system, but unlike Matlab,
is available at no cost to the end user.
Java is common enough that, for most, installing Nengo
is quite easy.

Nengo was a robust neural simulator, but required modelers
to be proficient in Java. To overcome this,
in the summer of 2008 (???), a graphical user interface
was created to make model creation and simulation
easy for anyone.

???YT link

The GUI leveraged the Java graphical toolkit Piccolo2D.java,
which makes it easy to make zoomable interfaces that
can play well with the Java Swing ecosystem.
The new GUI made it easy for beginners to become familiar
with the methods of the NEF, and gradually transition
to writing models outside of the GUI.

While Nengo and its GUI introduced many people
to the NEF, its deeply nested object hierarchy
proved difficult for many people to use productively.
While the GUI provided easy access for beginners,
the transition to being an expert user was difficult.
Additionally, while Java is cross platform and free to download,
it is not open source (though an open source version exists).
And while Java can simulate many networks quickly,
efforts to leverage non-standard computing devices
like GPUs were difficult to implement
in a cross-platform manner.

Important publications that used Nengo (Java):

* cite3
* cite4

Generation 3
============

Shortly after the GUI was released,
Terry Stewart began making it possible
to make simulations using Nengo
through a simple Python scripting interface.
The interface was originally known as ``nef.py``
due to the first implementation's file name,
but it quickly became the preferred way
for modelers to create models due to its simplicity,
and therefore was incorporated with the rest of Nengo.
While the scripting interface used Python syntax,
it was still able to operate within the Java ecosystem
thanks for a Java implementation of Python called Jython.

really productive

made spaun, really crazy

people wanted to make other large scale models,
but it's slow

because of that, started making a project
that used the nef.py scripts, but instead
of creating objects using Nengo (Java),
did it its own way using Theano

However, there were concerns that the Jython
and the Theano backed implementations would soon
diverge, fracturing the population of people building
nengo models

Generation 4
============

In order to deal with the fracturing of the Nengo community,
the decision was made to standardize the API that
had been evolving since the introduction of ``nef.py``.
Because the name "Nengo" was now well known,
the name stuck, and the API was called the Nengo API.

Through a grueling weekend of meetings,
the CNRG tentatively decided on an API
that any software claiming to be "Nengo"
would have to implement. In addition to the API,
the CNRG would produce a reference implementation
in Python with as few dependencies as possible,
change the Jython version to conform to the new API,
and in Theano to continue the work making a fast backend.

This generation is currently pushing forward!
This documentation lives in the same repository
as the Nengo API, which is still in some flux,
but will soon become a standard.
The reference implementation is moving forward at ???link.
The Jython version continues development at ???link.
The Theano version is being developed at ???link.

Although these three implementations may choose to
implement their own specific capabilities,
since they all conform to the Nengo API,
they can all run the vast majority of models
that a modeler would want to run.

We hope that, in this generation,
we have made all the right compromises such that
we can build large models with concise, expressive code,
and that we can create backends that can build and simulate
those models much more quickly than before.
Further, by making this API available,
we hope to be able to interact even further with
the rest of the neuroscience packages written in Python.

If you'd like to contribute to the development of Nengo,
please take a look one of the repositories below
and look at the list of issues to see what remains to be implemented!

Generation 5
============

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
-----------

Objects instead of strings
^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

No Origins and Terminations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
---------------------------

Many commonly used functions have been
simplified or changed to be more explicit.

Making ensembles
^^^^^^^^^^^^^^^^

Old API signature::

  nef.Network.make(name, neurons, dimensions, tau_rc, tau_ref, max_rate, intercept, radius, encoders, decoder_noise, eval_points, noise, noise_frequency, mode, add_to_network, node_factory, decoder_sign, seed, quick, storage_code)

A simple example::

  nef.Network.make('A', 40, 1, mode='spike')

TODO New API signature


TODO A simple example

See `Ensemble documentation <user_api.html#ensemble>`_
for a list of properties that can be manipulated.

Making ensemble arrays (i.e., network arrays)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Network arrays were very tightly coupled
with the old API. In the new API,
they have been decoupled and are now
an easily imported network instead.
The functionality should still be identical,
though the syntax has changed.

Old API::

  nef.Network.make_array(name, neurons, length, dimensions, **args)

New API::

  nengo.networks.EnsembleArray(name, neurons, n_ensembles, dimensions_per_ensemble, **ens_args)

See `EnsembleArray documentation <networks.html#ensemblearray>`_
for more information.

Making nodes
^^^^^^^^^^^^

Previously, there were several different ways
to provide input to a Nengo model:
``SimpleNode``, ``FunctionInput``, and others.
All of these use cases should be covered
by :class:`nengo.Node`.

In the old API, you could create your own
``SimpleNode``, or create a ``FunctionInput`` with::

  nef.Network.make_input(name, values, zero_after_time)

In the new API, you create a node with::

  nengo.Node(output)

where ``output`` is either a constant value
(float, list, NumPy array), a function, or
``None`` when passing through values unchanged.

See `Node documentation <user_api.html#node>`_
for more information.

Connecting things
^^^^^^^^^^^^^^^^^

A lot of the complexity of the old API
has been pushed down to the constructors
of the connection object.
In general, old API calls of the form::

  nef.Network.connect(pre, post)

are now::

  nengo.Connection(pre, post)

However, there are some changes in the additional arguments.
The old API used ``weight``, ``index_pre`` and ``index_post``
as a shortcut to define ``transform``;
in the new API, only the ``transform`` can be specified.
There are many NumPy functions that make transforms
easier to specify.
Additionally, we now utilize Python's slice syntax
to route dimensions easily::

  nengo.Connection(pre_1d, post_2d[0])

The keyword argument ``pstc`` has been renamed to ``synapse``.

Under the hood changes
----------------------

Under the hood, Nengo has been completely rewritten.
If you want to know the underlying structure of
Nengo 2.0, see the `developer documentation <dev_guide.html>`_.
