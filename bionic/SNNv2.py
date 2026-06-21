import pyrtl
from LIF_neuron import lif_neuron
from STDP_synapse import stdp_synapse

pyrtl.reset_working_block()

# Parameters
CAMERA_SIZE = 12
LIF_ARRAY_SIZE = 10
KERNEL_SIZE = 3 # 3x3 receptive field
W_BITWIDTH = 8
T_BITWIDTH = 5

# 1. 12x12 Camera Matrix (144 individual input pins)
camera_matrix = []
for y in range(CAMERA_SIZE):
    row = [pyrtl.Input(1, f"cam_y{y}_x{x}") for x in range(CAMERA_SIZE)]
    camera_matrix.append(row)

# 2. 10x10 Phosphene Output & Global Feedback Matrix
global_post_spike_matrix = []
phosphene_output_matrix = []
for y in range(LIF_ARRAY_SIZE):
    spike_row = []
    out_row = []
    for x in range(LIF_ARRAY_SIZE):
        # This wire carries the spike from the LIF neuron BACK to the synapses for STDP
        spike_wire = pyrtl.WireVector(1, name=f"post_spike_y{y}_x{x}")
        
        # This is the actual pin that connects to the patient's optic nerve interface
        out_wire = pyrtl.Output(1, name=f"phosphene_y{y}_x{x}")
        out_wire <<= spike_wire 
        
        spike_row.append(spike_wire)
        out_row.append(out_wire)
        
    global_post_spike_matrix.append(spike_row)
    phosphene_output_matrix.append(out_row)

# 3. Hardware Unrolling: 100 Neurons, 900 Synapses
for y in range(LIF_ARRAY_SIZE):
    for x in range(LIF_ARRAY_SIZE):
        
        synapse_currents = []
        
        # Solder 9 STDP synapses to this specific neuron
        for ky in range(KERNEL_SIZE):
            for kx in range(KERNEL_SIZE):
                
                # Calculate absolute camera coordinate
                cam_y = y + ky
                cam_x = x + kx
                
                # Instantiate 1 STDP synapse for this pixel-to-neuron connection
                current = stdp_synapse(
                    pre_spike=camera_matrix[cam_y][cam_x],
                    post_spike=global_post_spike_matrix[y][x], 
                    weight_bitwidth=W_BITWIDTH,
                    timer_bitwidth=T_BITWIDTH,
                    prefix=f"syn_y{y}x{x}_cam_y{cam_y}x{cam_x}"
                )
                synapse_currents.append(current)
        
        # FIX: Hardware Optimization (Balanced Adder Tree)
        # Python's built-in sum() is intercepted by PyRTL to build a logarithmic tree
        total_current = sum(synapse_currents)
        
        # Instantiate the localized LIF Neuron and wire its output to the feedback matrix
        global_post_spike_matrix[y][x] <<= lif_neuron(
            total_incoming_current=total_current,
            leak=5, 
            threshold=100, 
            bitwidth=16,
            prefix=f"lif_y{y}_x{x}"
        )

print("Hardware generated")
print(f"Total logic gates physically placed: {len(pyrtl.working_block().logic)}")