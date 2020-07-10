import inspect

from nengo.utils.magic import (
    decorator,
    ObjectProxyMethods,
    ObjectProxy,
    BoundFunctionWrapper,
)

state = None  # Used to make sure decorators are running


def _test_decorated(obj):
    global state

    # Make sure decorator works
    state = "not run"
    obj(0, 0)
    assert state == "run"
    state = "not run"

    # Make sure decorated function looks like non-decorated
    assert obj.__name__ == "f"
    assert obj.__doc__ == "Return 1."


def test_function():
    """Test that @decorator works on plain functions."""

    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = "run"
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
        global state
        state = "run"
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
        global state
        state = "run"
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
        global state
        state = "run"
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
        global state
        state = "run"
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


def test_decorator():
    class Test:
        instance = "Not None"

        @decorator
        def myMethod(self):
            pass

    # TODO: debug some stuff about the decorator
    # since lines 263-265 aren't being run
    # wrapped = Test()
    # wrapped.myMethod()
    # Can't figure out how to get instance to be None


def test_objectproxymethods():
    """tests functions of ObjectProxyMethods"""
    my_obj = ObjectProxyMethods()

    class Test:
        __module__ = None
        __dict__ = None

    my_obj.__wrapped__ = Test()

    test = my_obj.__module__
    assert test is None

    test = my_obj.__dict__
    assert test is None


def test_objectproxy():
    """tests functions of ObjectProxy"""

    class Test:
        __module__ = None
        __dict__ = None
        test = 1

    my_proxy = ObjectProxy(Test())
    assert (
        str(dir(my_proxy))
        == "['__class__', '__delattr__', '__dict__', '__dir__', '__doc__',"
        " '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__',"
        " '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',"
        " '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',"
        " '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',"
        " '__weakref__', 'test']"
        or str(dir(my_proxy))
        == "['__class__', '__delattr__', '__dict__', '__dir__', '__doc__',"
        " '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__',"
        " '__hash__', '__init__', '__le__', '__lt__',"
        " '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__',"
        " '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__',"
        " '__weakref__', 'test']"  # Travis-ci has this one
    )

    my_proxy = ObjectProxy(Test())
    assert hash(my_proxy) == hash(my_proxy)

    assert my_proxy.test == 1
    my_proxy.__wrapped__.test += 1
    my_proxy.__wrapped__ = None  # there is a slots pylint error when trying to set
    # an attribute not named __wrapped__
    assert my_proxy.test == 2
    assert my_proxy.__wrapped__.__wrapped__ is None

    instance = ObjectProxy(1)
    assert str(instance) == "1"

    assert repr(instance).startswith("<ObjectProxy at ")


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
