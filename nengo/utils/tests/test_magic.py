import inspect

from nengo.utils.magic import (
    decorator,
    ObjectProxy,
    BoundFunctionWrapper,
)


class RunState:
    state = None  # Used to make sure decorators are running


def _test_decorated(obj):

    # Make sure decorator works
    RunState.state = "not run"
    obj(0, 0)
    assert RunState.state == "run"
    RunState.state = "not run"

    # Make sure decorated function looks like non-decorated
    assert obj.__name__ == "f"
    assert obj.__doc__ == "Return 1."


def test_function():
    """Test that @decorator works on plain functions."""

    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        RunState.state = "run"
        assert instance is None
        return wrapped(*args, **kwargs)

    @test_decorator
    def f(a, b):
        """Return 1."""
        return 1

    _test_decorated(f)

    # Make sure introspection works
    assert inspect.getfullargspec(f).args == ["a", "b"]
    assert inspect.getsource(f) == (
        "    @test_decorator\n"
        "    def f(a, b):\n"
        '        """Return 1."""\n'
        "        return 1\n"
    )


def test_boundfunction():
    """Test that @decorator works on bound functions."""

    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        RunState.state = "run"
        assert instance is not None
        assert type(instance).__name__ == "Test"
        return wrapped(*args, **kwargs)

    class Test:
        @test_decorator
        def f(self, a, b):
            """Return 1."""
            return 1

    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["self", "a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @test_decorator\n"
        "        def f(self, a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )


def test_staticmethod():
    """Test that @decorator works on staticmethods."""

    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        RunState.state = "run"
        assert instance is None
        return wrapped(*args, **kwargs)

    # --- Decorator before staticmethod
    class TestBeforeStaticmethod:
        @test_decorator
        @staticmethod
        def f(a, b):
            """Return 1."""
            return 1

    _test_decorated(TestBeforeStaticmethod.f)
    inst = TestBeforeStaticmethod()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @test_decorator\n"
        "        @staticmethod\n"
        "        def f(a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )

    # --- Decorator after staticmethod
    class TestAfterStaticmethod:
        @staticmethod
        @test_decorator
        def f(a, b):
            """Return 1."""
            return 1

    _test_decorated(TestAfterStaticmethod.f)
    inst = TestAfterStaticmethod()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @staticmethod\n"
        "        @test_decorator\n"
        "        def f(a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )


def test_classmethod():
    """Test that @decorator works on classmethods."""

    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        RunState.state = "run"
        valid_names = {"TestBeforeStaticmethod", "TestAfterStaticmethod"}
        assert (
            instance is None
            and args[0].__name__ in valid_names
            or instance.__name__ in valid_names
        )
        return wrapped(*args, **kwargs)

    # --- Decorator before classmethod
    class TestBeforeStaticmethod:
        @test_decorator
        @classmethod
        def f(cls, a, b):
            """Return 1."""
            return 1

    _test_decorated(TestBeforeStaticmethod.f)
    inst = TestBeforeStaticmethod()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["cls", "a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @test_decorator\n"
        "        @classmethod\n"
        "        def f(cls, a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )

    # --- Decorator after staticmethod
    class TestAfterStaticmethod:
        @classmethod
        @test_decorator
        def f(cls, a, b):
            """Return 1."""
            return 1

    _test_decorated(TestAfterStaticmethod.f)
    inst = TestAfterStaticmethod()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["cls", "a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @classmethod\n"
        "        @test_decorator\n"
        "        def f(cls, a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )


def test_class():
    """Test that @decorator works on classes."""

    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        RunState.state = "run"
        assert instance is None
        inst = wrapped(*args, **kwargs)
        inst.ran = True
        return inst

    @test_decorator
    class f:
        """Return 1."""

        def __init__(self, a, b):
            self.a = a
            self.b = b

    _test_decorated(f)
    inst = f("a", "b")
    assert inst.a == "a" and inst.b == "b"
    assert inst.ran
    assert type(inst) == f.__wrapped__
    assert type(inst) == f.__wrapped__

    # Make sure introspection works
    # Note: for classes, the decorator isn't part of the source. Weird!
    assert inspect.getsource(f) == (
        "    class f:\n"
        '        """Return 1."""\n'
        "\n"
        "        def __init__(self, a, b):\n"
        "            self.a = a\n"
        "            self.b = b\n"
    )


def test_class_decorator():
    """Test that `decorator` works on a class method"""

    class TestA:
        @decorator
        @classmethod
        def test_decorator(cls, wrapped, instance, args, kwargs):
            RunState.state = "run"
            assert instance is not None
            assert type(instance).__name__ == "Test"
            return wrapped(*args, **kwargs)

    class Test:
        @TestA.test_decorator
        def f(self, a, b):
            """Return 1."""
            return 1

    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["self", "a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @TestA.test_decorator\n"
        "        def f(self, a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )


def test_instance_decorator():
    """Test that `decorator` works on a class instance method"""

    class TestA:
        @decorator
        def test_decorator(self, wrapped, instance, args, kwargs):
            RunState.state = "run"
            assert instance is not None
            assert type(instance).__name__ == "Test"
            return wrapped(*args, **kwargs)

    test_a = TestA()

    class Test:
        @test_a.test_decorator
        def f(self, a, b):
            """Return 1."""
            return 1

    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ["self", "a", "b"]
    assert inspect.getsource(inst.f) == (
        "        @test_a.test_decorator\n"
        "        def f(self, a, b):\n"
        '            """Return 1."""\n'
        "            return 1\n"
    )


def test_objectproxy():
    """tests functions of ObjectProxy"""

    class Test:
        """my docstring"""

        __annotations__ = "testannotations"
        __name__ = "testname"

        testvar = "testval"

    obj = Test()
    proxy = ObjectProxy(obj)

    # --- test ObjectProxyMethods
    assert proxy.__dict__ is obj.__dict__
    assert proxy.__doc__ is obj.__doc__
    assert proxy.__module__ is obj.__module__

    # --- test ObjectProxy
    assert proxy.__annotations__ is obj.__annotations__
    assert proxy.__name__ is obj.__name__
    assert proxy.__class__ is obj.__class__

    assert set(dir(proxy)) == set(dir(obj))
    assert hash(proxy) == hash(obj)
    assert str(proxy) == str(obj)

    proxy.newattr = "newval"  # pylint: disable=assigning-non-slot
    assert obj.newattr == "newval"
    assert proxy.newattr == "newval"

    assert repr(proxy).startswith("<ObjectProxy at ")


def test_boundfunctionwrapper():
    """tests functions of BoundFunctionWrapper"""

    class MyParentHelper:
        def __get__(self, a, b=None):
            return 0

    class MyParent:
        __wrapped__ = MyParentHelper

    class MyWrapped:
        value = 1

    class MyFunction:
        def __init__(self, b, c, d, e):
            return None

        def __call__(self, b):
            return True

    function = MyFunction

    wrapped = MyWrapped
    parent = MyParent

    instance = BoundFunctionWrapper(wrapped, None, function, "function", parent)

    assert (
        str(BoundFunctionWrapper.__get__(instance, instance, BoundFunctionWrapper))
        == "0"
    )

    instance2 = BoundFunctionWrapper(wrapped, "Not None", 1, "function", parent)

    assert (
        str(BoundFunctionWrapper.__get__(instance2, instance2, BoundFunctionWrapper))
        == "<class 'nengo.utils.tests.test_magic"
        ".test_boundfunctionwrapper.<locals>.MyWrapped'>"
    )

    assert str(instance(1, 2)).startswith(
        "<nengo.utils.tests.test_magic."
        "test_boundfunctionwrapper.<locals>.MyFunction object at "
    )
