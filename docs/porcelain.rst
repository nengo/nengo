Basic API
=========

The Nengo API operates on two levels.
This page describes the basic high-level interface
for creating neural simulations using the Nengo.

The easiest way to interact with Nengo
is to instantiate a :class:`nengo.nef.Network` object,
and call the appropriate methods on that object
to construct ensembles and connect them together.
This API should be sufficient for 90% of models
created using Nengo.

nengo.nef.Network
-----------

.. autoclass:: nengo.nef.Network
   :members:
