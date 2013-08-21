"""
Demonstrates how to create a neuronal ensemble that acts as an oscillator.

In this example, we examine a 2-D ring oscillator. The function an integrator
implements can be written in the following control theoretic equation:

  a_dot(t) = A_matrix * a(t)

where A_matrix = [ 0 w]
                 [-w 0]

The NEF equivalent A_matrix for this oscillator is: A_matrix = [ 1 1]
                                                               [-1 1]

The input to the network is needed to kick the system into
its stable oscillatory state.

Network diagram:
                    .----.
                    v    |
      [Input] ---> (A) --'

Network behaviour:
   A = A_matrix * A
"""

import nengo.old_api as api
import numpy as np

### Define model parameters
speed = 10       # Base frequency of oscillation
tau = 0.1        # TODO: this is supposed to be the feedback time constant

### Create the nengo model
model = api.Network('Integrator')

### Create the model inputs
def start_input(t):
    if t < 0.01: return [1,0]
    else:        return [0,0]

model.make_input('Input', start_input)

def speed_func(t):
    if   t < 0.3: return 1
    elif t < 0.6: return 0.5
    else:         return 1

model.make_input('Speed', speed_func)

### Create the neuronal ensemble
model.make('A', 500, 3, radius=1.7)

### Create the connections within the model
model.connect('Input', 'A', transform=[[1,0],[0,1],[0,0]])
model.connect('Speed', 'A', transform=[[0],[0],[1]])

def controlled_path(x):
    return [x[0] + x[2] * speed * tau * x[1],
            x[1] - x[2] * speed * tau * x[0],
            0]

# model.connect('A', 'A', func=controlled_path, pstc=tau)
model.connect('A', 'A', func=controlled_path)

### Add probes
probe_dt = 0.002
probe_tau = 0.03
input_p = model.make_probe('Input', 0.001, 0.001)
output_p = model.make_probe('A', probe_dt, probe_tau)

### Run the model
t_final = 1
model.run(t_final)

### Plot the results
try:
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    ins = input_p.get_data()
    outs = output_p.get_data()
    t = lambda x: (t_final/float(len(x)))*np.arange(len(x))

    plt.subplot(211)
    plt.plot(t(ins), ins)
    plt.subplot(212)
    plt.plot(t(outs), outs)
    # plt.show()

    ### animation
    import matplotlib.animation as animation

    x, y, w = outs.T

    fig = plt.figure(2)
    fig.clf()
    r = 1.5
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-r,r), ylim=(-r,r))
    ax.grid()

    dots = [ax.plot([],[],'ko-',markersize=(10-i)) for i in xrange(9)]
    dots = map(lambda x: x[0], dots)
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def init():
        for dot in dots:
            dot.set_data([],[])
        time_text.set_text('')
        return dots + [time_text]

    def animate(i):
        for j in xrange(0, min(i,len(dots))):
            dots[j].set_data([x[i-j]],[y[i-j]])
        time_text.set_text('time = %.3f' % (i*probe_dt))
        return dots + [time_text]

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, len(x)),
                                  interval=20, blit=True, init_func=init)

    plt.show()

except ImportError:
    print "Could not import required libraries for plotting"
