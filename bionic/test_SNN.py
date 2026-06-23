import pyrtl
from LIF_neuron import lif_neuron
from STDP_synapse import stdp_synapse

# 1. Reset the workspace
pyrtl.reset_working_block()

NUM_PIXELS = 4
WEIGHT_BITWIDTH = 8
TIMER_BITWIDTH = 5

# 2. Create the Inputs (mock camera pixels)
camera_pixels = [pyrtl.Input(1, f"pixel_{i}") for i in range(NUM_PIXELS)]

# 3. Forward Declaration: Global Feedback Wire
# Declare the wire now so we can hand it to the synapses, but we assign its logic later.
global_post_spike = pyrtl.WireVector(1, name="global_post_spike")

# 4. Instantiate the Synapses
synapse_currents = []
for i in range(NUM_PIXELS):
    current_out = stdp_synapse(pre_spike=camera_pixels[i], 
                               post_spike=global_post_spike, 
                               weight_bitwidth=WEIGHT_BITWIDTH, 
                               timer_bitwidth=TIMER_BITWIDTH,
                               prefix=f"pixel_{i}_syn")
    synapse_currents.append(current_out)

# 5. The Adder Tree (Summing the currents)
# PyRTL handles Python's sum() natively by generating a tree of hardware Adders!
total_current = sum(synapse_currents)

# 6. Instantiate the LIF Neuron & Close the Loop
# We use the <<= operator to connect the LIF's output BACK into our forward-declared wire.
global_post_spike <<= lif_neuron(
    total_incoming_current=total_current, 
    leak=2, 
    threshold=40, # High threshold so we can test accumulation
    bitwidth=16   # 16 bits so the sum of many 8-bit synapses doesn't overflow!
)

# Tell PyRTL this wire is our final network output
phosphene_output = pyrtl.Output(1, "phosphene_output") 
phosphene_output <<= global_post_spike

# --- SIMULATION SETUP ---

# 1. Grab the current working block (the silicon blueprint)
block = pyrtl.working_block()

# 2. Pre-load offline PyTorch baseline weights
initial_memory = {
    block.wirevector_by_name['pixel_0_syn_weight']: 15,
    block.wirevector_by_name['pixel_1_syn_weight']: 15,
    block.wirevector_by_name['pixel_2_syn_weight']: 15,
    block.wirevector_by_name['pixel_3_syn_weight']: 15,

    block.wirevector_by_name['pixel_0_syn_timer']: 20,
    block.wirevector_by_name['pixel_1_syn_timer']: 20,
    block.wirevector_by_name['pixel_2_syn_timer']: 20,
    block.wirevector_by_name['pixel_3_syn_timer']: 20
}

# 3. Create the simulator and inject the initial memory
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace, register_value_map=initial_memory)

# We will run the hardware for 15 clock cycles
for cycle in range(15):
    
    # Mock Camera Data: Pixels 0 and 1 fire on cycles 2, 3, and 6. 
    # Pixels 2 and 3 never fire.
    if cycle in [2, 3, 7]:
        inputs = {'pixel_0': 1, 'pixel_1': 1, 'pixel_2': 0, 'pixel_3': 0}
    else:
        inputs = {'pixel_0': 0, 'pixel_1': 0, 'pixel_2': 0, 'pixel_3': 0}
        
    sim.step(inputs)

print("--- PyRTL Hardware Simulation Trace ---")
sim_trace.render_trace()