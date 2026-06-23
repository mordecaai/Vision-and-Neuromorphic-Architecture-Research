import pyrtl

def lif_neuron(excitatory_current: pyrtl.WireVector, inhibitory_current: pyrtl.WireVector, leak: int, threshold: int, bitwidth: int = 8, prefix: str = "") -> pyrtl.WireVector:
    """
    A Leaky Integrate-and-Fire (LIF) Neuron acting as a post-synaptic current bucket.
    
    Architectural Note: Upstream STDP synapses now act as independent valves. If a pre-synaptic camera 
    pixel fires, its synapse outputs its specific weight; if it does not fire, 
    the synapse outputs 0. This neuron simply accepts the sum of those outputs.
    
    :param total_incoming_current: A dynamic WireVector representing the absolute sum of all 
                                   weighted currents from firing pre-synaptic synapses in the 
                                   current clock cycle.
    :param leak: Integer representing the passive voltage subtracted every clock cycle 
                 to simulate biological current leakage.
    :param threshold: Integer representing the voltage boundary required to trigger an action 
                      potential (output spike).
    :param bitwidth: Size of the internal voltage register. Must be large enough to hold 
                     the threshold value and prevent overflow from large current spikes.
    :return: 1-bit WireVector representing the output spike (1 if fired, 0 if quiet).
    """
    
    # 1. Membrane Voltage Accumulating and Serving as Memory (Capacitor)
    v_mem = pyrtl.Register(bitwidth=bitwidth, name=f'{prefix}_v_mem')

    # 2. Leak (Track remaining voltage in capacitor after passive leak is applied)
    remaining_v = pyrtl.select(v_mem >= leak, v_mem - leak, pyrtl.Const(0, bitwidth))

    # 3. Integration (The Input Current)
    # Add the  sum of all upstream synapses to our remaining voltage.
    gross_v = remaining_v + excitatory_current
    
    # HARDWARE SAFETY: Prevent unsigned underflow by flooring at 0
    will_underflow = inhibitory_current >= gross_v
    integrated_v = pyrtl.select(will_underflow, pyrtl.Const(0, bitwidth), gross_v - inhibitory_current)

    # 4. Dictate spike
    spike_out = pyrtl.WireVector(1, name=f"{prefix}_spike_out")
    spike_out <<= integrated_v >= pyrtl.Const(threshold, bitwidth) 

    # 5. Reset (Hard Reset to Zero)
    v_mem.next <<= pyrtl.select(spike_out, pyrtl.Const(0, bitwidth), integrated_v)

    return spike_out