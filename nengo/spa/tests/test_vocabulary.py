# -*- coding: utf-8 -*-

import numpy as np
import pytest

from nengo.exceptions import NengoWarning, SpaParseError, ValidationError
from nengo.spa import Vocabulary, VocabularyMap
from nengo.spa.pointer import Identity
from nengo.utils.testing import warns


def test_add(rng):
    v = Vocabulary(3, rng=rng)
    v.add('A', [1, 2, 3])
    v.add('B', [4, 5, 6])
    v.add('C', [7, 8, 9])
    assert np.allclose(v.vectors, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_populate(rng):
    v = Vocabulary(64, rng=rng)

    v.populate('A')
    assert 'A' in v

    v.populate('B; C')
    assert 'B' in v
    assert 'C' in v

    v.populate('D.unitary()')
    assert 'D' in v
    np.testing.assert_almost_equal(np.linalg.norm(v['D'].v), 1.)
    np.testing.assert_almost_equal(np.linalg.norm((v['D'] * v['D']).v), 1.)

    v.populate('E = A + 2 * B')
    assert np.allclose(v['E'].v, v.parse('A + 2 * B').v)
    assert np.linalg.norm(v['E'].v) > 2.

    v.populate('F = (A + 2 * B).normalized()')
    assert np.allclose(v['F'].v, v.parse('A + 2 * B').normalized().v)
    np.testing.assert_almost_equal(np.linalg.norm(v['F'].v), 1.)

    v.populate('G = A; H')
    assert np.allclose(v['G'].v, v['A'].v)
    assert 'H' in v

    # Assigning non-existing pointer
    with pytest.raises(NameError):
        v.populate('I = J')

    # Redefining
    with pytest.raises(ValidationError):
        v.populate('H = A')

    # Calling non existing function
    with pytest.raises(AttributeError):
        v.populate('I = H.invalid()')

    # invalid names: lowercase, unicode
    with pytest.raises(SpaParseError):
        v.populate('x = A')
    # with pytest.raises(SpaParseError):
    v.populate(u'Aα = A')


def test_parse(rng):
    v = Vocabulary(64, rng=rng)
    v.populate('A; B; C')
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
    with pytest.raises(SpaParseError):
        v.parse('"hello"')


def test_parse_n(rng):
    v = Vocabulary(64, rng=rng)
    v.populate('A; B; C')
    A = v.parse('A')
    B = v.parse('B')

    parsed = v.parse_n('A', 'A*B', 'A+B', '3')
    assert np.allclose(parsed[0].v, A.v)
    assert np.allclose(parsed[1].v, (A * B).v)
    assert np.allclose(parsed[2].v, (A + B).v)
    assert np.allclose(parsed[3].v, 3 * Identity(64).v)


def test_invalid_dimensions():
    with pytest.raises(ValidationError):
        Vocabulary(1.5)
    with pytest.raises(ValidationError):
        Vocabulary(0)
    with pytest.raises(ValidationError):
        Vocabulary(-1)


def test_capital(rng):
    v = Vocabulary(16, rng=rng)
    with pytest.raises(SpaParseError):
        v.parse('a')
    with pytest.raises(SpaParseError):
        v.parse('A+B+C+a')


def test_transform(recwarn, rng):
    v1 = Vocabulary(32, strict=False, rng=rng)
    v2 = Vocabulary(64, strict=False, rng=rng)
    v1.populate('A; B; C')
    v2.populate('A; B; C')
    A = v1['A']
    B = v1['B']
    C = v1['C']

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

    # Test warns on missing keys
    v1.populate('D')
    D = v1['D']
    with warns(NengoWarning):
        v1.transform_to(v2)

    # Test populating missing keys
    t = v1.transform_to(v2, populate=True)
    assert v2.parse('D').compare(np.dot(t, D.v)) > 0.95

    # Test ignores missing keys in source vocab
    v2.populate('E')
    v1.transform_to(v2, populate=True)
    assert 'E' not in v1


def test_create_pointer_warning(rng):
    v = Vocabulary(2, rng=rng)

    # five pointers shouldn't fit
    with warns(UserWarning):
        v.populate('A; B; C; D; E')


def test_readonly(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.populate('A;B;C')

    v1.readonly = True

    with pytest.raises(ValueError):
        v1.parse('D')


def test_subset(rng):
    v1 = Vocabulary(32, rng=rng)
    v1.populate('A; B; C; D; E; F; G')

    # Test creating a vocabulary subset
    v2 = v1.create_subset(['A', 'C', 'E'])
    assert list(v2.keys()) == ['A', 'C', 'E']
    assert v2['A'] == v1['A']
    assert v2['C'] == v1['C']
    assert v2['E'] == v1['E']


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
