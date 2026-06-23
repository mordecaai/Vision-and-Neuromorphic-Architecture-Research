import pyrtl
from LIF_neuron import lif_neuron

pyrtl.reset_working_block()

# --- Hardware Parameters ---
CAMERA_SIZE = 12
LIF_ARRAY_SIZE = 10
KERNEL_SIZE = 3 

# The Quantized 
WEIGHTS = [
    [-15, -15, -15],
    [-15, 120, -15],
    [-15, -15, -15]
]

# --- 1. Silicon Pins ---
# Camera Input Matrix (144 Pins)
camera_matrix = [[pyrtl.Input(1, f"cam_y{y}_x{x}") for x in range(CAMERA_SIZE)] for y in range(CAMERA_SIZE)]

# Phosphene Output Matrix (100 Pins)
phosphene_matrix = [[pyrtl.Output(1, f"phosphene_y{y}_x{x}") for x in range(LIF_ARRAY_SIZE)] for y in range(LIF_ARRAY_SIZE)]

# --- 2. Hardware Unrolling (Static Inference) ---
for y in range(LIF_ARRAY_SIZE):
    for x in range(LIF_ARRAY_SIZE):
        
        # Dale's Principle: Separate Positive (Excitatory) and Negative (Inhibitory) wires
        excitatory_currents = []
        inhibitory_currents = []
        
        # Solder the 9 pixels directly to the mathematically perfect weights
        for ky in range(KERNEL_SIZE):
            for kx in range(KERNEL_SIZE):
                cam_y = y + ky
                cam_x = x + kx
                weight_val = WEIGHTS[ky][kx]
                
                # Hardware MUX: If pixel=1, output weight magnitude, else 0.
                
                # Excitatory Routing (Positive Weights)
                if weight_val > 0:
                    current = pyrtl.select(
                        camera_matrix[cam_y][cam_x], 
                        pyrtl.Const(weight_val, bitwidth=8), 
                        pyrtl.Const(0, bitwidth=8)
                    )
                    excitatory_currents.append(current)
                
                # Inhibitory Routing (Negative Weights -> Converted to positive magnitude drain)
                elif weight_val < 0:
                    current = pyrtl.select(
                        camera_matrix[cam_y][cam_x], 
                        pyrtl.Const(abs(weight_val), bitwidth=8), 
                        pyrtl.Const(0, bitwidth=8)
                    )
                    inhibitory_currents.append(current)
                
        # Two Separate Logarithmic Adder Trees
        # (No Two's Complement math needed! All wires are unsigned positive integers)
        total_exc = sum(excitatory_currents)
        total_inh = sum(inhibitory_currents)
        
        # The localized LIF Neuron (Dual-Path)
        phosphene_matrix[y][x] <<= lif_neuron(
            excitatory_current=total_exc,
            inhibitory_current=total_inh,
            leak=5, 
            threshold=85, 
            bitwidth=12,
            prefix=f"lif_y{y}_x{x}"
        )

print("Hardware successfully generated!")
print(f"Total logic gates physically placed: {len(pyrtl.working_block().logic)}")