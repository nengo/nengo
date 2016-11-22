import re

import numpy as np
import pytest

from nengo.exceptions import SpaParseError, ValidationError
from nengo.spa import Vocabulary, VocabularyMap
from nengo.utils.testing import warns


def test_add(rng):
    v = Vocabulary(3, rng=rng)
    v.add('A', [1, 2, 3])
    v.add('B', [4, 5, 6])
    v.add('C', [7, 8, 9])
    assert np.allclose(v.vectors, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_include_pairs(rng):
    v = Vocabulary(10, rng=rng)
    v.populate('A, B, C')
    assert v.key_pairs is None
    v.include_pairs = True
    assert v.key_pairs == ['A*B', 'A*C', 'B*C']
    v.include_pairs = False
    assert v.key_pairs is None
    v.include_pairs = True
    v.populate('D')
    assert v.key_pairs == ['A*B', 'A*C', 'B*C', 'A*D', 'B*D', 'C*D']

    v = Vocabulary(12, include_pairs=True, rng=rng)
    v.populate('A, B, C')
    assert v.key_pairs == ['A*B', 'A*C', 'B*C']


def test_populate(rng):
    v = Vocabulary(64, rng=rng)

    v.populate('A')
    assert 'A' in v

    v.populate('B, C')
    assert 'B' in v
    assert 'C' in v

    v.populate('D.unitary()')
    assert 'D' in v
    np.testing.assert_almost_equal(np.linalg.norm(v['D'].v), 1.)
    np.testing.assert_almost_equal(np.linalg.norm((v['D'] * v['D']).v), 1.)

    v.populate('E = A + 2 * B')
    assert np.allclose(v['E'].v, v.parse('A + 2 * B').v)
    assert np.linalg.norm(v['E'].v) > 2.

    v.populate('F = (A + 2 * B).normalized()')  # FIXME SPs are always normalized
    assert np.allclose(v['F'].v, v.parse('A + 2 * B').normalized().v)
    np.testing.assert_almost_equal(np.linalg.norm(v['F'].v), 1.)

    v.populate('G = A, H')
    assert np.allclose(v['G'].v, v['A'].v)
    assert 'H' in v

    # TODO test non-SP error cases


def test_parse(rng):
    v = Vocabulary(64, rng=rng)
    v.populate('A, B, C')
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
    v.populate('A, B, C, D, E, F')
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


# def test_transform(rng):
    # v1 = Vocabulary(32, rng=rng)
    # v2 = Vocabulary(64, rng=rng)
    # v1.populate('A, B, C')
    # A = v1['A']
    # B = v1['B']
    # C = v1['C']

    # # Test transform from v1 to v2 (full vocbulary)
    # # Expected: np.dot(t, A.v) ~= v2.parse('A')
    # # Expected: np.dot(t, B.v) ~= v2.parse('B')
    # # Expected: np.dot(t, C.v) ~= v2.parse('C')
    # t = v1.transform_to(v2)

    # assert v2.parse('A').compare(np.dot(t, A.v)) > 0.95
    # assert v2.parse('C+B').compare(np.dot(t, C.v + B.v)) > 0.9

    # # Test transform from v1 to v2 (only 'A' and 'B')
    # t = v1.transform_to(v2, keys=['A', 'B'])

    # assert v2.parse('A').compare(np.dot(t, A.v)) > 0.95
    # assert v2.parse('B').compare(np.dot(t, C.v + B.v)) > 0.95

    # # Test transform_to when either vocabulary is read-only
    # v1.parse('D')
    # v2.parse('E')

    # # When both are read-only, transform_to shouldn't add any new items to
    # # either and the transform should be using keys that are the intersection
    # # of both vocabularies
    # v1.readonly = True
    # v2.readonly = True

    # t = v1.transform_to(v2)

    # assert v1.keys == ['A', 'B', 'C', 'D']
    # assert v2.keys == ['A', 'B', 'C', 'E']

    # # When one is read-only, transform_to should add any new items to the non
    # # read-only vocabulary
    # v1.readonly = False
    # v2.readonly = True

    # t = v1.transform_to(v2)

    # assert v1.keys == ['A', 'B', 'C', 'D', 'E']
    # assert v2.keys == ['A', 'B', 'C', 'E']

    # # When one is read-only, transform_to should add any new items to the non
    # # read-only vocabulary
    # v1.readonly = True
    # v2.readonly = False

    # t = v1.transform_to(v2)

    # assert v1.keys == ['A', 'B', 'C', 'D', 'E']
    # assert v2.keys == ['A', 'B', 'C', 'E', 'D']


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
    with warns(UserWarning):
        v.populate('A, B, C, D, E')


def test_readonly(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.populate('A,B,C')

    v1.readonly = True

    with pytest.raises(ValueError):
        v1.parse('D')


def test_subset(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.populate('A, B, C, D, E, F, G')

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


def test_vocabulary_set(rng):
    v8 = Vocabulary(8)
    v16 = Vocabulary(16)
    v32 = Vocabulary(32)
    vs = VocabularyMap([v8, v16], rng=rng)

    # Behaviour common to set and dict
    assert len(vs) == 2
    assert 8 in vs
    assert 16 in vs
    assert 32 not in vs

    assert v8 in vs
    assert v16 in vs
    assert v32 not in vs
    assert Vocabulary(8) not in vs

    # dict behaviour
    assert vs[8] is v8
    assert vs[16] is v16

    del vs[8]
    assert 8 not in vs

    # set behaviour
    vs.add(v32)
    assert vs[32] is v32
    with pytest.warns(UserWarning):
        vs.add(v32)

    vs.discard(32)
    assert 32 not in vs
    vs.discard(v16)
    assert 16 not in vs

    # creating new vocabs if non existent
    vs.add(v8)
    assert vs.get_or_create(8) is v8
    new = vs.get_or_create(16)
    assert vs[16] is new
    assert new.dimensions == 16
    assert new.rng is rng
