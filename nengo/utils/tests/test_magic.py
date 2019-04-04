import inspect

from nengo.utils.magic import decorator

state = None  # Used to make sure decorators are running


def _test_decorated(obj):
    global state

    # Make sure decorator works
    state = 'not run'
    obj(0, 0)
    assert state == 'run'
    state = 'not run'

    # Make sure decorated function looks like non-decorated
    assert obj.__name__ == 'f'
    assert obj.__doc__ == "Return 1."


def test_function():
    """Test that @decorator works on plain functions."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert instance is None
        return wrapped(*args, **kwargs)

    @test_decorator
    def f(a, b):
        """Return 1."""
        return 1

    _test_decorated(f)

    # Make sure introspection works
    assert inspect.getfullargspec(f).args == ['a', 'b']
    assert inspect.getsource(f) == ('    @test_decorator\n'
                                    '    def f(a, b):\n'
                                    '        """Return 1."""\n'
                                    '        return 1\n')


def test_boundfunction():
    """Test that @decorator works on bound functions."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        assert instance is not None
        assert type(instance).__name__ == 'Test'
        return wrapped(*args, **kwargs)

    class Test:
        @test_decorator
        def f(self, a, b):
            """Return 1."""
            return 1

    inst = Test()
    _test_decorated(inst.f)

    # Make sure introspection works
    assert inspect.getfullargspec(inst.f).args == ['self', 'a', 'b']
    assert inspect.getsource(inst.f) == ('        @test_decorator\n'
                                         '        def f(self, a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')


def test_staticmethod():
    """Test that @decorator works on staticmethods."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
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
    assert inspect.getfullargspec(inst.f).args == ['a', 'b']
    assert inspect.getsource(inst.f) == ('        @test_decorator\n'
                                         '        @staticmethod\n'
                                         '        def f(a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')

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
    assert inspect.getfullargspec(inst.f).args == ['a', 'b']
    assert inspect.getsource(inst.f) == ('        @staticmethod\n'
                                         '        @test_decorator\n'
                                         '        def f(a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')


def test_classmethod():
    """Test that @decorator works on classmethods."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
        valid_names = {'TestBeforeStaticmethod', 'TestAfterStaticmethod'}
        assert (instance is None and args[0].__name__ in valid_names
                or instance.__name__ in valid_names)
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
    assert inspect.getfullargspec(inst.f).args == ['cls', 'a', 'b']
    assert inspect.getsource(inst.f) == ('        @test_decorator\n'
                                         '        @classmethod\n'
                                         '        def f(cls, a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')

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
    assert inspect.getfullargspec(inst.f).args == ['cls', 'a', 'b']
    assert inspect.getsource(inst.f) == ('        @classmethod\n'
                                         '        @test_decorator\n'
                                         '        def f(cls, a, b):\n'
                                         '            """Return 1."""\n'
                                         '            return 1\n')


def test_class():
    """Test that @decorator works on classes."""
    @decorator
    def test_decorator(wrapped, instance, args, kwargs):
        global state
        state = 'run'
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
    inst = f('a', 'b')
    assert inst.a == 'a' and inst.b == 'b'
    assert inst.ran
    assert type(inst) == f.__wrapped__
    assert type(inst) == f.__wrapped__

    # Make sure introspection works
    # Note: for classes, the decorator isn't part of the source. Weird!
    assert inspect.getsource(f) == ('    class f:\n'
                                    '        """Return 1."""\n'
                                    '        def __init__(self, a, b):\n'
                                    '            self.a = a\n'
                                    '            self.b = b\n')
