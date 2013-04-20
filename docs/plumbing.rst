============
Advanced API
============

The Nengo API operates on two levels.
This page describes the more complicated low-level interface
for creating neural simulations using Nengo.

This API is designed for more experienced
modelers who need more complicated functionality
than is offered by the :class:`nengo.nef.Network` class.
This API exposes the underlying objects
that are created by the methods in :class:`nengo.nef.Network`,
allowing for more fine-grained control and subclassing.

