#
# Notes on hacking.
# First attempt: subclass Encoder
# (a) that requires changing scripts rather than improving models,
# (b) related: that approach doesn't work for networks
#
# Second attempt: walk a model and install customized encoders.
#
# 

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

import nengo
import nengo.helpers
from nengo.networks.oscillator import Oscillator
from nengo.tests.helpers import Plotter, rmse, SimulatorTestCase, unittest

def targets(conn, eval_points):
    if conn.function is None:
        return eval_points
    else:
        rval = np.array(
            [conn.function(ep) for ep in eval_points])
        if len(rval.shape) < 2:
            rval.shape = rval.shape[0], 1
        return rval


def lif_rates_drates(ens, J_without_bias, bias):
    old = np.seterr(divide='ignore', invalid='ignore')
    try:
        J = J_without_bias + bias
        tau_ref = ens.neurons.tau_ref
        tau_rc = ens.neurons.tau_rc

        t0 = J ** 2
        t1 = (1 - 1.0 / np.maximum(J, 0))
        logt1 = np.log(t1)

        A = tau_ref - tau_rc * logt1
        Ainv = 1 / A
        J1 = J > 1
        act = np.where(J1, Ainv, 0)
        dJ = np.where(J1, tau_rc / (t0 * t1 * logt1 ** 2), 0)
        return act, dJ
    finally:
        np.seterr(**old)


def pack(*args):
    return np.hstack([a.ravel() for a in args])

def unpack(theta, args):
    offset = 0
    rval = []
    for a in args:
        rval.append(theta[offset:offset + a.size].reshape(a.shape))
        offset += a.size
    return rval

def encoders_by_backprop(ens, dt):
    print 'figuring out encoders for', ens

    # Set up neurons
    # XXX should ideally not do this until after the deepcopy
    #     has been made in simulator construction
    if ens.neurons.gain is None:
        ens.set_neuron_properties()

    # Set up encoder
    encoders = nengo.decoders.sample_hypersphere(
        ens.dimensions, ens.neurons.n_neurons,
        ens.rng, surface=True)
    encoders /= np.asarray(ens.radius)
    encoders *= ens.neurons.gain[:, np.newaxis]
    print 'radius', ens.radius

    # Find out how this ensemble is used.
    users = [obj for obj in ens.connections_out
        if isinstance(obj, nengo.connections.DecodedConnection)]
    print 'users:', ens.connections_out
    if len(users) == 0:
        return

    # Determine on what points to train the encoders
    eval_points = np.vstack([user.eval_points for user in users])
    targ_points = np.hstack([targets(user, eval_points) for user in users])
    assert eval_points.ndim == 2
    assert targ_points.ndim == 2
    assert len(eval_points) == len(targ_points)

    print eval_points.shape, encoders.shape
    if not isinstance(ens.neurons, nengo.core._LIFBase):
        print 'skipping non-LIF ensemble', ens
        return

    J_nobias = np.dot(eval_points, encoders.T)
    activities = ens.neurons.rates(J_nobias)
    decoders, res, rank, s = np.linalg.lstsq(
        activities, targ_points, rcond=.01)

    theta0 = pack(encoders, decoders, ens.neurons.bias)

    ## L2-DECAY is hyperparam
    def func(theta, l2_penalty=0.0001):
        enc, dec, bb = unpack(theta, (encoders, decoders, ens.neurons.bias))

        J_nobias = np.dot(eval_points, enc.T)
        act, dJ = lif_rates_drates(ens, J_nobias, bb)
        pred = np.dot(act, dec)
        err = pred - targ_points
        err_cost = 0.5 * np.sum(err ** 2)
        l2_cost = 0.5 * np.sum(decoders ** 2)
        cost = err_cost + l2_penalty * l2_cost
        #print 'sse', sse, dJ.max()
        dact = np.dot(err, dec.T)
        dJ *= dact
        dbias = dJ.sum(axis=0)
        ddec = np.dot(act.T, err) + l2_penalty * dec
        denc = np.dot(dJ.T, eval_points)
        return cost, pack(denc, ddec, dbias)

    theta_opt, _, _ = fmin_l_bfgs_b(
        func=func,
        x0=theta0,
        maxfun=50,
        iprint=2,
        )


class TestOscillator(SimulatorTestCase):
    def test_oscillator(self):
        model = nengo.Model('Oscillator')
        inputs = {0:[1,0],0.5:[0,0]}
        model.make_node('Input', nengo.helpers.piecewise(inputs))

        tau = 0.1
        freq = 5
        model.add(Oscillator('T', tau, freq, neurons=nengo.LIF(100)))
        model.connect('Input', 'T')

        model.make_ensemble('A', nengo.LIF(100), dimensions=2)
        model.connect('A', 'A', transform=[[1, -freq*tau], [freq*tau, 1]],
                      filter=tau)
        model.connect('Input', 'A')

        model.probe('Input')
        model.probe('A', filter=0.01)
        model.probe('T', filter=0.01)

        if 1:
          for obj in model.objs.values():
            if isinstance(obj, nengo.objects.Ensemble) and obj.encoders is None:
                encoders_by_backprop(obj, dt=0.001)

        sim = model.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(3.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(model.t)
            plt.plot(t, sim.data('A'), label='Manual')
            plt.plot(t, sim.data('T'), label='Template')
            plt.plot(t, sim.data('Input'), 'k', label='Input')
            plt.legend(loc=0)
            plt.savefig('test_oscillator.test_oscillator_bp.pdf')
            plt.close()
            
        self.assertTrue(rmse(sim.data('A'), sim.data('T')) < 0.3)

if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()

