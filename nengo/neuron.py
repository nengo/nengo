import numpy as np


def lif_step(J, voltage, refractory_time, spiked, dt, tau_rc, tau_ref, upsample):
    if upsample != 1:
        raise NotImplementedError()

    # Euler's method
    dV = dt / tau_rc * (J - voltage)

    # increase the voltage, ignore values below 0
    v = np.maximum(voltage + dV, 0)  
    
    # handle refractory period        
    post_ref = 1.0 - (refractory_time - dt) / dt

    # set any post_ref elements < 0 = 0, and > 1 = 1
    v *= np.clip(post_ref, 0, 1)
    
    # determine which neurons spike
    # if v > 1 set spiked = 1, else 0
    spiked[:] = (v > 1) * 1.0
    
    # adjust refractory time (neurons that spike get
    # a new refractory time set, all others get it reduced by dt)

    # linearly approximate time since neuron crossed spike threshold
    overshoot = (v - 1) / dV 
    spiketime = dt * (1.0 - overshoot)

    # adjust refractory time (neurons that spike get a new
    # refractory time set, all others get it reduced by dt)
    new_refractory_time = spiked * (spiketime + tau_ref) \
            + (1 - spiked) * (refractory_time - dt)

    # return an ordered dictionary of internal variables to update
    # (including setting a neuron that spikes to a voltage of 0)

    voltage[:] = v * (1 - spiked)
    refractory_time[:] = new_refractory_time

def batch_lif_rates(J, tau_rc, tau_ref):
    A = tau_ref - tau_rc * np.log(1 - 1.0 / np.maximum(J, 0))
    
    # if input current is enough to make neuron spike,
    # calculate firing rate, else return 0
    A = np.where(J > 1, 1 / A, 0)
    return A

