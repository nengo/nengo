import numpy as np
import pytest

import nengo


def test_io(tmpdir):
    tmpfile = str(tmpdir.join("model.pkl"))
    m1 = nengo.Model("test_io")
    with m1:
        sin = nengo.Node(output=np.sin)
        cons = nengo.Node(output=-.5)
        factors = nengo.Ensemble(nengo.LIF(20), dimensions=2, radius=1.5)
        factors.encoders = np.tile(
            [[1, 1], [-1, 1], [1, -1], [-1, -1]],
            (factors.n_neurons // 4, 1))
        product = nengo.Ensemble(nengo.LIFRate(10), dimensions=1)
        nengo.Connection(sin, factors[0])
        nengo.Connection(cons, factors[1])
        factors_p = nengo.Probe(
            factors, 'decoded_output', sample_every=.01, filter=.01)
        assert factors_p  # To suppress F841
        product_p = nengo.Probe(
            product, 'decoded_output', sample_every=.01, filter=.01)
        assert product_p  # To suppress F841
    m1.save(tmpfile)
    m2 = nengo.Model.load(tmpfile)
    assert m1 == m2


if __name__ == "__main__":
    nengo.log(debug=True)
    pytest.main([__file__, '-v'])
