import pytest

from nengo.utils.testing import ThreadedAssertion
from nengo.utils.threading import ThreadLocalStack


class TestThreadLocalStack:
    def test_threadsafety(self):
        stack = ThreadLocalStack()
        stack.append(1)

        class CheckIndependence(ThreadedAssertion):
            def init_thread(self, worker):
                stack.append(2)

            def assert_thread(self, worker):
                assert list(stack) == [2]

        CheckIndependence(n_threads=2)

    def test_has_length(self):
        stack = ThreadLocalStack()
        assert len(stack) == 0
        stack.append(1)
        assert len(stack) == 1

    def test_implements_stack(self):
        stack = ThreadLocalStack()
        assert list(stack) == []
        stack.append(1)
        assert list(stack) == [1]
        stack.append(2)
        assert list(stack) == [1, 2]
        assert stack.pop() == 2
        assert list(stack) == [1]
        assert stack.pop() == 1
        assert list(stack) == []

        with pytest.raises(IndexError):
            stack.pop()

    def test_implements_clear(self):
        stack = ThreadLocalStack()
        stack.append(1)
        stack.clear()
        assert len(stack) == 0

    def test_has_size_limit(self):
        maxsize = 5
        stack = ThreadLocalStack(maxsize=maxsize)
        for i in range(maxsize):
            stack.append(i)

        with pytest.raises(RuntimeError):
            stack.append(5)
