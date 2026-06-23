import pyrtl
import static_SNN

# 1. Simulator
sim_trace = pyrtl.SimulationTrace()
sim = pyrtl.Simulation(tracer=sim_trace)

print("Starting Simulation")

# 2. Generate the Stimulus (A vertical line moving left to right)
NUM_FRAMES = 10

for frame_idx in range(NUM_FRAMES):
    # Create a blank 12x12 camera frame (all 0s)
    current_inputs = {}
    for y in range(static_SNN.CAMERA_SIZE):
        for x in range(static_SNN.CAMERA_SIZE):
            pin_name = f"cam_y{y}_x{x}"
            current_inputs[pin_name] = 0
            
    # Draw the vertical line at the current frame index
    line_x_position = frame_idx
    for y in range(static_SNN.CAMERA_SIZE):
        pin_name = f"cam_y{y}_x{line_x_position}"
        current_inputs[pin_name] = 1
        
    # 3. Step Hardware Clock
    sim.step(current_inputs)

print("Simulation Complete")

# 4. Render the Oscilloscope Waveform
sim_trace.render_trace(trace_list=[f"phosphene_y0_x{x}" for x in range(10)])