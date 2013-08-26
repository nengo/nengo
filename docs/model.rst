===========
Basic usage
===========

The easiest way to use Nengo is to use the ``Model``.
Everything that you need
for building most simple models can be done with
a ``Model``.

To create a new ``Model``::

  import nengo
  model = nengo.Model("My Model")

Creating Nengo objects
======================

A Nengo object is a part of your model that represents information.
There are two objects that make up a basic Nengo model.
Each of these objects can be constructed by calling
the appropriate ``make_`` function.

An ``Ensemble`` is a group of neurons that represents
information in the form of real valued numbers.

::

  my_ensemble = model.make_ensemble("My Ensemble", nengo.LEIF(35), dimensions=1)

In this case, My Ensemble is made up of
35 leaky integrate-and-fire neurons,
and it is representing a one dimensional signal.
In other words, this ensemble represents a single number.

In order to provide input to this ensemble,
to emulate some signal that exists in nature, for example,
we create a ``Node``.

::

  my_node = model.make_node("My Node", output=0.5)

In this case, My Node emits the number 0.5.

In most cases, however, we want more dynamic information.
We can make a ``Node`` using a function as output
instead of a number.

::

  import math
  sin_node = model.make_node("Sine Node"), output=math.sin)

This node will represent the sine over time.

Connecting Nengo objects
========================

We can connect our nodes to ensembles
in order to represent that information
in the activity a group of neurons.

::

  model.connect(my_node, my_ensemble)

The connect My Node to My Ensemble,
meaning that My Ensemble will now represent
0.5 in its population of 35 neurons.

This works because the dimensionality of
``my_node`` and ``my_ensemble`` is the same.
If this were not true, we would
define the transformation between
the dimensions of the first object to the second object
with a ``transform`` argument.
For example::

  two_d_ensemble = model.make_ensemble("2D Ensemble", nengo.LIF(70), dimensions=2)
  model.connect(sin_node, two_d_ensemble, transform=[[1], [0]])

This creates a new ensemble that represents
two real-valued signals.
By connecting the Sine Node to the ensemble,
its first dimension now represents
a sine wave.

Ensembles can be connected to each other as well.
When connecting ensembles together,
we can optionally compute an arbitrary smooth function
on the real numbers being connected.

::

  model.connect(my_ensemble, two_d_ensemble, transform=[[0], [1]], function=math.sqrt)

Functions can be computed over multiple dimensions, as well.

::

  product_ensemble = model.make_ensemble("Product", nengo.LIF(35), dimensions=1)
  def product(x):
      return x[0] * x[1]
  model.connect(two_d_ensemble, product_ensemble, function=product)

Probing Nengo objects
=====================

Once you have defined the objects if your model
and how they're connected,
you have to decide what data you want to collect
by probing those objects.

For example, if we wanted to collect data from
our 2D Ensemble and the Product of those two dimensions::

  model.probe(two_d_ensemble, filter=0.01)
  model.probe(product_ensemble, filter=0.01)

The argument ``filter`` defines the time constant
on a causal low-pass filter.
The output of ensembles of spiking neurons
can be very noisy, so a filter is recommended.

Running an experiment
=====================

Once a model has been constructed and we have probed
certain objects, we can run it to collect data.

To run a model, we must first build a simulator
based on the model we've defined.

::

  sim = model.simulator()

We can then run that simulator.
For example, to run our model for five seconds::

  sim.run(5.0)

One a simulation has been run at least once
(it can be run for additional time if desired)
the data collected can be accessed
for analysis or visualization.

::

  product_data = sim.data(product_ensemble)

For more details on the functions
in ``nengo.Model``, and ``nengo.simulator.Simulator``
see :ref:`basic-api`.

========
Examples
========

Representing information
========================

.. toctree::

   A single neurons <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/singleneuron.ipynb>
   Two neurons <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/twoneurons.ipynb>
   Many neurons <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/manyneurons.ipynb>
   2D representation <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/2drepresentation.ipynb>


Transforming information
========================

.. toctree::

   Communication channel <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/singleneuron.ipynb>
   Squaring the input <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/squaring.ipynb>
   Addition <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/addition.ipynb>
   Combining 1D representations into a 2D representation <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/combining.ipynb>
   Multiplication <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/multiplication.ipynb>
   Circular convolution <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/convolution.ipynb>

Dynamics
========

.. toctree::

   Simple integrator <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/integrator.ipynb>
   Controlled integrator <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/controlledintegrator.ipynb>
   Controlled integrator 2 <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/controlledintegrator2.ipynb>
   Simple harmonic oscillator <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/oscillator.ipynb>
   Controlled oscillator <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/controlledoscillator.ipynb>

Learning
========

.. toctree::

   Learning a communication channel <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/learn_communicationchannel.ipynb>
   Learning multiplication <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/learn_multiplication.ipynb>
   Learning element-wise square <http://nbviewer.ipython.org/urls/raw.github.com/ctn-waterloo/nengo/master/examples/learn_square.ipynb>
