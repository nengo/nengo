import numpy as np
import pytest

from nengo.spa.pointer import SemanticPointer
from nengo.utils.compat import range


def test_init():
    a = SemanticPointer([1, 2, 3, 4])
    assert len(a) == 4

    a = SemanticPointer([1, 2, 3, 4, 5])
    assert len(a) == 5

    a = SemanticPointer(list(range(100)))
    assert len(a) == 100

    a = SemanticPointer(27)
    assert len(a) == 27
    assert np.allclose(a.length(), 1)

    with pytest.raises(Exception):
        a = SemanticPointer(np.zeros(2, 2))

    with pytest.raises(Exception):
        a = SemanticPointer(-1)
    with pytest.raises(Exception):
        a = SemanticPointer(0)
    with pytest.raises(Exception):
        a = SemanticPointer(1.7)
    with pytest.raises(Exception):
        a = SemanticPointer(None)
    with pytest.raises(Exception):
        a = SemanticPointer(int)


def test_length():
    a = SemanticPointer([1, 1])
    assert np.allclose(a.length(), np.sqrt(2))
    a = SemanticPointer(10)*1.2
    assert np.allclose(a.length(), 1.2)


def test_normalize():
    a = SemanticPointer([1, 1])
    a.normalize()
    assert np.allclose(a.length(), 1)


def test_str():
    a = SemanticPointer([1, 1])
    assert str(a) == '[ 1.  1.]'


def test_randomize(rng):
    a = SemanticPointer(100, rng=rng)
    std = np.std(a.v)
    assert np.allclose(std, 1.0 / np.sqrt(len(a)), rtol=0.02)

    a = SemanticPointer(100)
    a.randomize(rng=rng)
    assert len(a) == 100
    std = np.std(a.v)
    assert np.allclose(std, 1.0 / np.sqrt(len(a)), rtol=0.02)

    a = SemanticPointer(5)
    a.randomize(N=100, rng=rng)
    assert len(a) == 100
    std = np.std(a.v)
    assert np.allclose(std, 1.0 / np.sqrt(len(a)), rtol=0.02)


def test_make_unitary(rng):
    a = SemanticPointer(100, rng=rng).unitary()
    assert np.allclose(1, a.length())
    assert np.allclose(1, (a * a).length())
    assert np.allclose(1, (a * a * a).length())


def test_add_sub():
    a = SemanticPointer(10)
    b = SemanticPointer(10)
    c = a.copy()
    d = b.copy()

    c += b
    d -= -a

    assert np.allclose((a + b).v, a.v + b.v)
    assert np.allclose((a + b).v, c.v)
    assert np.allclose((a + b).v, d.v)
    assert np.allclose((a + b).v, (a - (-b)).v)


def test_convolution(rng):
    a = SemanticPointer(64, rng=rng)
    b = SemanticPointer(64, rng=rng)
    identity = SemanticPointer(np.eye(64)[0])

    c = a.copy()
    c *= b

    ans = np.fft.ifft(np.fft.fft(a.v) * np.fft.fft(b.v)).real

    assert np.allclose((a * b).v, ans)
    assert np.allclose(a.convolve(b).v, ans)
    assert np.allclose(c.v, ans)
    assert np.allclose((a * identity).v, a.v)
    assert (a * b * ~b).compare(a) > 0.65


def test_multiply():
    a = SemanticPointer(50)

    assert np.allclose((a * 5).v, a.v * 5)
    assert np.allclose((5 * a).v, a.v * 5)
    assert np.allclose((a * 5.7).v, a.v * 5.7)
    assert np.allclose((5.7 * a).v, a.v * 5.7)
    assert np.allclose((0 * a).v, np.zeros(50))
    assert np.allclose((1 * a).v, a.v)

    with pytest.raises(Exception):
        a * None
    with pytest.raises(Exception):
        a * 'string'


def test_compare(rng):
    a = SemanticPointer(50, rng=rng) * 10
    b = SemanticPointer(50, rng=rng) * 0.1

    assert a.compare(a) > 0.99
    assert a.compare(b) < 0.2
    assert np.allclose(a.compare(b), a.dot(b) / (a.length() * b.length()))


def test_dot(rng):
    a = SemanticPointer(50, rng=rng) * 1.1
    b = SemanticPointer(50, rng=rng) * (-1.5)
    assert np.allclose(a.dot(b), np.dot(a.v, b.v))


def test_distance(rng):
    a = SemanticPointer(50, rng=rng)
    b = SemanticPointer(50, rng=rng)
    assert a.distance(a) < 1e-5
    assert a.distance(b) > 0.9


def test_invert():
    a = SemanticPointer(50)
    assert a.v[0] == (~a).v[0]
    assert a.v[49] == (~a).v[1]
    assert np.allclose(a.v[1:], (~a).v[:0:-1])


def test_len():
    a = SemanticPointer(5)
    assert len(a) == 5

    a = SemanticPointer(list(range(10)))
    assert len(a) == 10


def test_copy():
    a = SemanticPointer(5)
    b = a.copy()
    assert a is not b
    assert a.v is not b.v
    assert np.allclose(a.v, b.v)


def test_mse():
    a = SemanticPointer(50)
    b = SemanticPointer(50)

    assert np.allclose(((a - b).length() ** 2) / 50, a.mse(b))


def test_conv_matrix():
    a = SemanticPointer(50)
    b = SemanticPointer(50)

    m = b.get_convolution_matrix()

    assert np.allclose((a*b).v, np.dot(m, a.v))
