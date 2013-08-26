============
Core objects
============

These classes are used to describe a Nengo model to be simulated.
All other objects use describe models in terms of these objects.
Simulators only know about these objects.

TODO more

TODO block diagram?

Signals
=======

.. autoclass:: nengo.core.Signal
   :members:

.. autoclass:: nengo.core.SignalView
   :members:

.. autoclass:: nengo.core.Constant
   :members:

Linear transforms
=================

.. autoclass:: nengo.core.Transform
   :members:

.. autoclass:: nengo.core.Filter
   :members:

.. autoclass:: nengo.core.Encoder
   :members:

.. autoclass:: nengo.core.Decoder
   :members:

Nonlinearities
==============

.. autoclass:: nengo.core.Nonlinearity
   :members:

.. autoclass:: nengo.core.Direct
   :members:

.. autoclass:: nengo.core.LIF
   :members:

.. autoclass:: nengo.core.LIFRate
   :members:
