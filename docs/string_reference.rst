======================================
Referencing Nengo objects with strings
======================================

Every object in a Nengo model can be uniquely identified
with a string. If you'd like, you can always use the string
representation of an object instead of the object in
``Model.connect`` and ``Model.connect_neurons``.

To refer to an ensemble in your model, you can just use its name.
For example::

  model = nengo.Model("My model")
  model.make_ensemble("A", neurons=40, dimensions=1)
  model.make_ensemble("B", neurons=40, dimensions=1)
  model.connect("A", "B")

If an ensemble has been grouped into a network, use the
pattern ``network/`` to index into the network.
For example, ensemble ``A`` in the network ``My Network``
would referenced by the string ``My Network/A``.
In code, continuing from the above example::

  net = model.make_network("My Network")
  net.make_ensemble("A")
  model.connect("A", "My Network/A")

Note that an ensemble within a network can have the same name
as an ensemble that is not in that network.
Names must be unique within a network, however.

When probing a Nengo object, you can access a property
of that object using the ``.`` character.
For example, the membrane voltages of the ``A`` ensemble
would be accessed with ``A.voltages``.
In code, continuing from the above example::

  model.probe("A.voltages")
  model.probe("My Network/A.spikes")

To probe a property on the connection itself,
you must access the connection with a string of the form
``pre>post``. For example, the connections above would be referenced
by ``A>B`` and ``A>My Network/A`` respectively.
In code, this would be::

  model.probe("A>B.decoders")
  model.probe("A>My Network/A.weights")

Finally, two objects can be connected multiple times
if the functions connecting them are different.
In these cases, we can refer to a specific connection
with the form ``pre:func>post``.
For example::

  def square(x):
      return x * x
  model.connect("A", "B", func=square)
  model.probe("A:square>B")

Quick reference
===============

For connect, use the following format for the string::

  "net_1/.../net_n/object"

For probes, use the following format to probe objects::

  "net_1/.../net_n/object.property"

And the following format to probe connections::

  "net_1/.../object1>net_2/.../object2.property"  # most connections
  "net_1/.../object1:function1>net_2/.../object2.property"  # if necessary
