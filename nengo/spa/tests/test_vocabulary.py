import numpy as np
import pytest

import nengo
from nengo.spa import Vocabulary


def test_add():
    v = Vocabulary(3)
    v.add('a', [1, 2, 3])
    v.add('b', [4, 5, 6])
    v.add('c', [7, 8, 9])
    assert np.allclose(v.vectors, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_include_pairs():
    v = Vocabulary(10)
    v['a']
    v['b']
    v['c']
    assert v.key_pairs is None
    v.include_pairs = True
    assert v.key_pairs == ['a*b', 'a*c', 'b*c']
    v.include_pairs = False
    assert v.key_pairs is None
    v.include_pairs = True
    v['d']
    assert v.key_pairs == ['a*b', 'a*c', 'b*c', 'a*d', 'b*d', 'c*d']

    v = Vocabulary(12, include_pairs=True)
    v['a']
    v['b']
    v['c']
    assert v.key_pairs == ['a*b', 'a*c', 'b*c']


def test_parse():
    v = Vocabulary(64)
    a = v.parse('a')
    b = v.parse('b')
    c = v.parse('c')
    assert np.allclose((a * b).v, v.parse('a * b').v)
    assert np.allclose((a * ~b).v, v.parse('a * ~b').v)
    assert np.allclose((a + b).v, v.parse('a + b').v)
    assert np.allclose((a - (b*c)*3 + ~c).v, v.parse('a-(b*c)*3+~c').v)

    assert np.allclose(v.parse('0').v, np.zeros(64))
    assert np.allclose(v.parse('1').v, np.eye(64)[0])
    assert np.allclose(v.parse('1.7').v, np.eye(64)[0] * 1.7)

    with pytest.raises(SyntaxError):
        v.parse('a((')
    with pytest.raises(TypeError):
        v.parse('"hello"')


def test_identity():
    v = Vocabulary(64)
    assert np.allclose(v.identity.v, np.eye(64)[0])


def test_text():
    rng = np.random.RandomState(1)
    v = Vocabulary(64, rng=rng)
    x = v.parse('a+b+c')
    y = v.parse('-d-e-f')
    assert v.text(x) == '0.99a;0.96c;0.90b'
    assert v.text(x, maximum_count=2) == '0.99a;0.96c'
    assert v.text(x, maximum_count=1) == '0.99a'
    assert v.text(x, maximum_count=10) == '0.99a;0.96c;0.90b'
    assert v.text(x, minimum_count=4) == '0.99a;0.96c;0.90b;-0.02d'
    assert v.text(y) == '0.50c;0.15b'
    assert v.text(y, threshold=0.6) == '0.50c'
    assert v.text(y, minimum_count=None, threshold=0.6) == ''
    assert (v.text(x, minimum_count=4, terms=['a', 'b', 'c']) ==
            '0.99a;0.96c;0.90b')

    assert v.text(x, join=',') == '0.99a,0.96c,0.90b'
    assert v.text(x, normalize=True) == '0.59a;0.57c;0.53b'

    assert v.text([0]*64) == '0.00f'
    assert v.text(v['d'].v) == '1.00d'


def test_transform():
    v1 = Vocabulary(32, rng=np.random.RandomState(7))
    v2 = Vocabulary(64, rng=np.random.RandomState(8))
    a = v1.parse('a')
    b = v1.parse('b')
    c = v1.parse('c')
    t = v1.transform_to(v2)

    assert v2.parse('a').compare(np.dot(t, a.v)) > 0.95
    assert v2.parse('c+b').compare(np.dot(t, c.v + b.v)) > 0.95

    t = v1.transform_to(v2, keys=['a', 'b'])

    assert v2.parse('a').compare(np.dot(t, a.v)) > 0.95
    assert v2.parse('b').compare(np.dot(t, c.v + b.v)) > 0.95


def test_prob_cleanup():
    v = Vocabulary(64)
    assert 1.0 > v.prob_cleanup(0.7, 10000) > 0.9999
    assert 0.9999 > v.prob_cleanup(0.6, 10000) > 0.999
    assert 0.99 > v.prob_cleanup(0.5, 1000) > 0.9

    v = Vocabulary(128)
    assert 0.999 > v.prob_cleanup(0.4, 1000) > 0.997
    assert 0.99 > v.prob_cleanup(0.4, 10000) > 0.97
    assert 0.9 > v.prob_cleanup(0.4, 100000) > 0.8


if __name__ == '__main__':
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
