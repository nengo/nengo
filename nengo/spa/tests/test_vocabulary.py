import re

import numpy as np
import pytest

from nengo.exceptions import SpaParseError, ValidationError
from nengo.spa import Vocabulary


def test_add(rng):
    v = Vocabulary(3, rng=rng)
    v.add('A', [1, 2, 3])
    v.add('B', [4, 5, 6])
    v.add('C', [7, 8, 9])
    assert np.allclose(v.vectors, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_include_pairs(rng):
    v = Vocabulary(10, rng=rng)
    v['A']
    v['B']
    v['C']
    assert v.key_pairs is None
    v.include_pairs = True
    assert v.key_pairs == ['A*B', 'A*C', 'B*C']
    v.include_pairs = False
    assert v.key_pairs is None
    v.include_pairs = True
    v['D']
    assert v.key_pairs == ['A*B', 'A*C', 'B*C', 'A*D', 'B*D', 'C*D']

    v = Vocabulary(12, include_pairs=True, rng=rng)
    v['A']
    v['B']
    v['C']
    assert v.key_pairs == ['A*B', 'A*C', 'B*C']


def test_parse(rng):
    v = Vocabulary(64, rng=rng)
    A = v.parse('A')
    B = v.parse('B')
    C = v.parse('C')
    assert np.allclose((A * B).v, v.parse('A * B').v)
    assert np.allclose((A * ~B).v, v.parse('A * ~B').v)
    assert np.allclose((A + B).v, v.parse('A + B').v)
    assert np.allclose((A - (B*C)*3 + ~C).v, v.parse('A-(B*C)*3+~C').v)

    assert np.allclose(v.parse('0').v, np.zeros(64))
    assert np.allclose(v.parse('1').v, np.eye(64)[0])
    assert np.allclose(v.parse('1.7').v, np.eye(64)[0] * 1.7)

    with pytest.raises(SyntaxError):
        v.parse('A((')
    with pytest.raises(SpaParseError):
        v.parse('"hello"')


def test_invalid_dimensions():
    with pytest.raises(ValidationError):
        Vocabulary(1.5)
    with pytest.raises(ValidationError):
        Vocabulary(0)
    with pytest.raises(ValidationError):
        Vocabulary(-1)


def test_identity(rng):
    v = Vocabulary(64, rng=rng)
    assert np.allclose(v.identity.v, np.eye(64)[0])


def test_text(rng):
    v = Vocabulary(64, rng=rng)
    x = v.parse('A+B+C')
    y = v.parse('-D-E-F')
    ptr = r'-?[01]\.[0-9]{2}[A-F]'
    assert re.match(';'.join([ptr] * 3), v.text(x))
    assert re.match(';'.join([ptr] * 2), v.text(x, maximum_count=2))
    assert re.match(ptr, v.text(x, maximum_count=1))
    assert len(v.text(x, maximum_count=10).split(';')) <= 10
    assert re.match(';'.join([ptr] * 4), v.text(x, minimum_count=4))
    assert re.match(';'.join([ptr.replace('F', 'C')] * 3),
                    v.text(x, minimum_count=4, terms=['A', 'B', 'C']))

    assert re.match(ptr, v.text(y, threshold=0.6))
    assert v.text(y, minimum_count=None, threshold=0.6) == ''

    assert v.text(x, join=',') == v.text(x).replace(';', ',')
    assert re.match(';'.join([ptr] * 2), v.text(x, normalize=True))

    assert v.text([0]*64) == '0.00F'
    assert v.text(v['D'].v) == '1.00D'


def test_capital(rng):
    v = Vocabulary(16, rng=rng)
    with pytest.raises(SpaParseError):
        v.parse('a')
    with pytest.raises(SpaParseError):
        v.parse('A+B+C+a')


def test_transform(rng):
    v1 = Vocabulary(32, rng=rng)
    v2 = Vocabulary(64, rng=rng)
    A = v1.parse('A')
    B = v1.parse('B')
    C = v1.parse('C')

    # Test transform from v1 to v2 (full vocbulary)
    # Expected: np.dot(t, A.v) ~= v2.parse('A')
    # Expected: np.dot(t, B.v) ~= v2.parse('B')
    # Expected: np.dot(t, C.v) ~= v2.parse('C')
    t = v1.transform_to(v2)

    assert v2.parse('A').compare(np.dot(t, A.v)) > 0.95
    assert v2.parse('C+B').compare(np.dot(t, C.v + B.v)) > 0.9

    # Test transform from v1 to v2 (only 'A' and 'B')
    t = v1.transform_to(v2, keys=['A', 'B'])

    assert v2.parse('A').compare(np.dot(t, A.v)) > 0.95
    assert v2.parse('B').compare(np.dot(t, C.v + B.v)) > 0.95

    # Test transform_to when either vocabulary is read-only
    v1.parse('D')
    v2.parse('E')

    # When both are read-only, transform_to shouldn't add any new items to
    # either and the transform should be using keys that are the intersection
    # of both vocabularies
    v1.readonly = True
    v2.readonly = True

    t = v1.transform_to(v2)

    assert v1.keys == ['A', 'B', 'C', 'D']
    assert v2.keys == ['A', 'B', 'C', 'E']

    # When one is read-only, transform_to should add any new items to the non
    # read-only vocabulary
    v1.readonly = False
    v2.readonly = True

    t = v1.transform_to(v2)

    assert v1.keys == ['A', 'B', 'C', 'D', 'E']
    assert v2.keys == ['A', 'B', 'C', 'E']

    # When one is read-only, transform_to should add any new items to the non
    # read-only vocabulary
    v1.readonly = True
    v2.readonly = False

    t = v1.transform_to(v2)

    assert v1.keys == ['A', 'B', 'C', 'D', 'E']
    assert v2.keys == ['A', 'B', 'C', 'E', 'D']


def test_prob_cleanup(rng):
    v = Vocabulary(64, rng=rng)
    assert 1.0 > v.prob_cleanup(0.7, 10000) > 0.9999
    assert 0.9999 > v.prob_cleanup(0.6, 10000) > 0.999
    assert 0.99 > v.prob_cleanup(0.5, 1000) > 0.9

    v = Vocabulary(128, rng=rng)
    assert 0.999 > v.prob_cleanup(0.4, 1000) > 0.997
    assert 0.99 > v.prob_cleanup(0.4, 10000) > 0.97
    assert 0.9 > v.prob_cleanup(0.4, 100000) > 0.8


def test_create_pointer_warning(rng):
    v = Vocabulary(2, rng=rng)

    # five pointers shouldn't fit
    with pytest.warns(UserWarning):
        v.parse('A')
        v.parse('B')
        v.parse('C')
        v.parse('D')
        v.parse('E')


def test_readonly(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.parse('A+B+C')

    v1.readonly = True

    with pytest.raises(ValueError):
        v1.parse('D')


def test_subset(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.parse('A+B+C+D+E+F+G')

    # Test creating a vocabulary subset
    v2 = v1.create_subset(['A', 'C', 'E'])
    assert v2.keys == ['A', 'C', 'E']
    assert v2['A'] == v1['A']
    assert v2['C'] == v1['C']
    assert v2['E'] == v1['E']
    assert v2.parent is v1

    # Test creating a subset from a subset (it should create off the parent)
    v3 = v2.create_subset(['C', 'E'])
    assert v3.parent is v2.parent and v2.parent is v1

    v3.include_pairs = True
    assert v3.key_pairs == ['C*E']
    assert not v1.include_pairs
    assert not v2.include_pairs

    # Test transform_to between subsets (should be identity transform)
    t = v1.transform_to(v2)

    assert v2.parse('A').compare(np.dot(t, v1.parse('A').v)) >= 0.99999999


def test_extend(rng):
    v = Vocabulary(16, rng=rng)
    v.parse('A+B')
    assert v.keys == ['A', 'B']
    assert not v.unitary

    # Test extending the vocabulary
    v.extend(['C', 'D'])
    assert v.keys == ['A', 'B', 'C', 'D']

    # Test extending the vocabulary with various unitary options
    v.extend(['E', 'F'], unitary=['E'])
    assert v.keys == ['A', 'B', 'C', 'D', 'E', 'F']
    assert v.unitary == ['E']

    # Check if 'E' is unitary
    fft_val = np.fft.fft(v['E'].v)
    fft_imag = fft_val.imag
    fft_real = fft_val.real
    fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
    assert np.allclose(fft_norms, np.ones(16))

    v.extend(['G', 'H'], unitary=True)
    assert v.keys == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    assert v.unitary == ['E', 'G', 'H']
