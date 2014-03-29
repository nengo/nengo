from nengo.utils.decorators import decorator


@decorator
def with_self(method, network, args, kwargs):
    """Wraps a method with ``with network:``.

    This makes it easy to add methods to a network that create new
    Nengo objects. Instead of writing ``with self`` at the top of the method
    and indenting everything over, you can instead use this decorator.

    Example
    -------

    The two methods in the following class do the same thing::

        class MyNetwork(nengo.Network):
            def add_one_1(self):
                with self:
                    node = nengo.Node(output=1)

            @with_self
            def add_one_2(self):
                node = nengo.Node(output=1)
    """
    with network:
        return method(*args, **kwargs)
