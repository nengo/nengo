"""
This example demonstrates how to create an integrator in neurons.
  The function an integrator implements can be written in the
  following control theoretic equation:

    a_dot(t) = A * a(t) + B * input(t)

  The NEF equivalent equation for this integrator is:

    a_dot(t) = a(t) + tau * input(t)

  where tau is the recurrent time constant.

Network diagram:

                    .----.
                    v    |
     [Input] ----> (A) --'


Network behaviour:
   A = tau * Input + Input

"""

import nengo.old_api as nengo

### Define model parameters
tau = 0.1

### Create the nengo model
model = nengo.Network('Integrator')

### Create the model inputs
input_d = {0.2:5, 0.3:0, 0.44:-10, 0.54:0, 0.8:5, 0.9:0}.items()
input_d.sort(reverse=True)
def input_f(t):
    for key, value in input_d:
        if t > key: return value
    return 0.0

model.make_input('Input', input_f)
# model.make_input('Input', input_d)

### Create the neuronal ensembles
model.make('A', 100, 1, radius=1)

### Create the connections within the model
# model.connect('Input', 'A', transform=[tau], pstc=0.005)
model.connect('Input', 'A', transform=[tau], pstc=tau)
model.connect('A', 'A', pstc=tau)

### Add probes
probe_dt = 0.01
probe_tau = 0.03
input_p = model.make_probe('Input', 0.001, 0.001)
output_p = model.make_probe('A', probe_dt, probe_tau)

### Build the model
# model.build()

### Run the model
t_final = 1.2
model.run(t_final)

### Plot the results
try:
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    x = input_p.get_data()
    y = output_p.get_data()
    t = lambda x: (t_final/len(x))*np.arange(len(x))

    plt.subplot(211)
    plt.plot(t(x), x)
    plt.subplot(212)
    plt.plot(t(y), y)
    plt.show()
except ImportError:
    print "Could not import required libraries for plotting"
