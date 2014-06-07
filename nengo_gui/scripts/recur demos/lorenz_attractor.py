# ## The Lorenz chaotic attractor
# 
# This example shows the construction of a classic chaotic dynamical system: the Lorenz "butterfly" attractor.  The equations are:
#         
# $$
# \dot{x}_0 = \sigma(x_1 - x_0) \\\
# \dot{x}_1 = x_0 (\rho - x_2) - x_1  \\\
# \dot{x}_2 = x_0 x_1 - \beta x_2 
# $$
# 
# Since $x_2$ is centered around approximately $\rho$, and since NEF ensembles are usually optimized to represent values within a certain radius of the origin, we substitute $x_2' = x_2 - \rho$, giving these equations:
# $$
# \dot{x}_0 = \sigma(x_1 - x_0) \\\
# \dot{x}_1 = - x_0 x_2' - x_1\\\
# \dot{x}_2' = x_0 x_1 - \beta (x_2' + \rho) - \rho
# $$
# 
# For more information, see http://compneuro.uwaterloo.ca/publications/eliasmith2005b.html "Chris Eliasmith. A unified approach to building and controlling spiking attractor networks. Neural computation, 7(6):1276-1314, 2005."

tau = 0.1
sigma = 10
beta = 8.0/3
rho = 28

import nengo

def feedback(x):
    dx0 = -sigma * x[0] + sigma * x[1]
    dx1 = -x[0] * x[2] - x[1]
    dx2 = x[0] * x[1] - beta * (x[2] + rho) - rho
    
    return [dx0 * tau + x[0], 
            dx1 * tau + x[1], 
            dx2 * tau + x[2]]

model = nengo.Network(label='Lorenz attractor')
with model:
    state = nengo.Ensemble(2000, 3, radius=60, label="state")
    nengo.Connection(state, state, function=feedback, synapse=tau)
    state_probe = nengo.Probe(state, synapse=tau)



import nengo_gui
gui = nengo_gui.Config()
gui[model].scale = 2.6605652662574757
gui[model].offset = 379.08657696393794,153.60400337944498
gui[state].pos = 0.000, 0.000
gui[state].scale = 1.000
