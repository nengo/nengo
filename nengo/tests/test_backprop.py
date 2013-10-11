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
from nengo.tests.helpers import Plotter, SimulatorTestCase, unittest

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
        dJ = np.where(J1, np.minimum(200,
                              tau_rc / (t0 * t1 * (-tau_rc * logt1 + tau_ref) ** 2)), 0)
        return act, dJ
    finally:
        np.seterr(**old)


def pack(*args):
    return np.hstack([a.flatten() for a in args])


def unpack(theta, args):
    offset = 0
    rval = []
    for a in args:
        rval.append(theta[offset:offset + a.size].reshape(a.shape))
        offset += a.size
    return rval


@nengo.decoders.timer('encoders_by_backprop')
def encoders_by_backprop(ens, dt, l2_penalty=0.1, ):
    print 'figuring out encoders for', ens, 'l2', l2_penalty

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

    if not isinstance(ens.neurons, nengo.core._LIFBase):
        print 'skipping non-LIF ensemble', ens
        return

    J_nobias = np.dot(eval_points, encoders.T)
    activities = ens.neurons.rates(J_nobias)
    decoders, res, rank, s = np.linalg.lstsq(
        activities, targ_points, rcond=.01)

    verbose = activities.shape[1] == 50

    if verbose:
        print 'eval shape', eval_points.shape
        print 'eval range', eval_points.min(), eval_points.max()
        print 'act shape', activities.shape
        print 'act range', activities.min(), activities.mean(), activities.max()
        print 'dec shape', decoders.shape
        print 'targ shape', targ_points.shape

    # XXX add something to bias so that more activations are non-zero when
    # fitting encoders
    theta0 = pack(encoders, decoders, ens.neurons.bias)

    ## L2-DECAY is hyperparam
    def func(theta, train_dec=True):
        enc, dec, bb = unpack(theta, (encoders, decoders, ens.neurons.bias))

        J_nobias = np.dot(eval_points, enc.T)
        act, dJ = lif_rates_drates(ens, J_nobias, bb)
        if verbose:
            #import matplotlib.pyplot as plt
            #plt.scatter((J_nobias + bb).flatten(), act.flatten())
            #plt.scatter((J_nobias + bb).flatten(), dJ.flatten())
            #plt.show()
            print 'enc', enc.min(), enc.mean(), enc.max()
        #print (J_nobias + bb).min(), (J_nobias + bb).max()
        pred = np.dot(act, dec)
        err = pred - targ_points
        if verbose:
            print '  err', err.min(), err.mean(), err.max(), 'total', np.sum(err ** 2)
        err_cost = 0.5 * np.sum(err ** 2)
        l2_cost = 0.5 * np.sum(decoders ** 2)
        cost = err_cost + l2_penalty * l2_cost
        #print 'sse', sse, dJ.max()
        dact = np.dot(err, dec.T)
        
        #  print dec.shape
        if verbose:
            print 'dact', dact.min(), dact.mean(), dact.max()

        dJ *= dact
        TUNE_ENCODERS = 1
        dbias = dJ.sum(axis=0) * TUNE_ENCODERS
        if train_dec:
            ddec = np.dot(act.T, err) + l2_penalty * dec
        else:
            ddec = np.zeros_like(dec)
        denc = np.dot(dJ.T, eval_points) * TUNE_ENCODERS
        if verbose:
            print '    denc', denc.min(), denc.mean(), denc.max()
        return cost, pack(denc, ddec, dbias)

    if 1:
        theta_opt, _, _ = fmin_l_bfgs_b(
            func=func,
            x0=theta0,
            maxfun=100, # HYPER
            iprint=0,
            args=(False,),
            )
        theta_opt, _, _ = fmin_l_bfgs_b(
            func=func,
            x0=theta_opt,
            maxfun=50, # HYPER
            iprint=0,
            args=(True,),
            )
        print 'func theta_opt'
        func(theta_opt)
    else:
        func(theta0)
        theta_opt = theta0

    enc, dec, bb = unpack(theta_opt, (encoders, decoders, ens.neurons.bias))
    ens.encoders = enc
    ens.neurons.bias = bb
    # XXX Why is this necessary? Why doesn't the built-in decoder solver work very well?
    #assert len(users) == 1
    #users[0].decoders = dec / dt

class TestBackprop(SimulatorTestCase):
    def test_bp(self):
        model = nengo.Model('Oscillator', seed=123)
        tspeed = 1.0
        model.make_node('Input', lambda t: np.sin(tspeed * t))
        n_neurons = 16
        model.make_ensemble('A', nengo.LIF(n_neurons), dimensions=1, seed=123)
        model.make_ensemble('B', nengo.LIF(n_neurons), dimensions=1, seed=123)
        model.make_ensemble('AA', nengo.LIF(n_neurons * 3), dimensions=1, seed=3)
        model.make_ensemble('BB', nengo.LIF(n_neurons * 3), dimensions=1, seed=3)
        model.connect('Input', 'A')
        model.connect('Input', 'B')
        model.connect('A', 'AA', function=lambda x: x ** 2, filter=0.03,
                     decoder_solver=nengo.decoders.ridge_regression)
        model.connect('B', 'BB', function=lambda x: x ** 2, filter=0.03)
        #model.probe('A', filter=0.03)
        #model.probe('B', filter=0.03)
        model.probe('AA', filter=0.05)
        model.probe('BB', filter=0.05)

        if 1:
          for obj in model.objs.values():
            if isinstance(obj, nengo.objects.Ensemble) and obj.encoders is None:
                if obj.name.startswith("B"):
                    encoders_by_backprop(obj, dt=0.001)

        sim = model.simulator(dt=0.001, sim_class=self.Simulator)
        sim.run(10.0)

        with Plotter(self.Simulator) as plt:
            t = sim.data(model.t)
            plt.plot(t, sim.data('AA'), label='Random')
            plt.plot(t, sim.data('BB'), label='BP')
            plt.plot(t, np.sin(tspeed * t) ** 2, label='C')
            plt.legend(loc='center')
            plt.savefig('test_backprop.test_bp.pdf')
            plt.close()
            
        #self.assertTrue(rmse(sim.data('Input'), sim.data('A')) < 0.3)

if __name__ == "__main__":
    nengo.log_to_file('log.txt', debug=True)
    unittest.main()

