import nengo
import numpy as np

from hunse_tools.timing import tic, toc

nt = 10000
nx = 1000

X = np.random.uniform(-1, 1, size=(nt, nx))

noden = nengo.synapses.LinearFilter((1,), (1,), default_dt=0.001)
lowpass = nengo.synapses.Lowpass(0.01, default_dt=0.001)
alpha = nengo.synapses.Alpha(0.01, default_dt=0.001)

n = 5

ts = []
for _ in range(n):
    tic('Noden')
    Y1 = noden.filt(X)
    ts.append(toc(display=False))

print("Noden: %0.3f seconds" % min(ts))

ts = []
for _ in range(n):
    tic('Lowpass')
    Y1 = lowpass.filt(X)
    ts.append(toc(display=False))

print("Lowpass: %0.3f seconds" % min(ts))

ts = []
for _ in range(n):
    tic('Alpha')
    Y2 = alpha.filt(X)
    ts.append(toc(display=False))

print("Alpha: %0.3f seconds" % min(ts))

# master:
#  Lowpass: 0.077 seconds
#  Alpha: 0.177 seconds
