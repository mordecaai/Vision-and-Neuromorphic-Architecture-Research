import torch
import matplotlib.pyplot as plt

def generate_moving_vertical_edge(time_steps=10, height=12, width=12):
    """
    Generates a synthetic DVS tensor of a moving vertical line.
    Shape required by snnTorch: [time_steps, batch_size, channels, height, width]
    """
    # Initialize an empty tensor with all zeros (no spikes)
    # Batch size = 1, Channels = 1 (Grayscale)
    dvs_data = torch.zeros((time_steps, 1, 1, height, width))
    
    # Sweep a vertical line from left to right across the camera
    for t in range(time_steps):
        # To keep it within the 12x12 grid, we use modulo or just bound it
        col = t % width 
        
        # Set the entire column to '1' (Spike) for this specific time step
        dvs_data[t, 0, 0, :, col] = 1.0
        
    return dvs_data

# 1. Generate the toy data
synthetic_tensor = generate_moving_vertical_edge()

# 2. Verify the shape
print(f"Generated Tensor Shape: {synthetic_tensor.shape}")
# Expected Output: torch.Size([10, 1, 1, 12, 12])

# Optional: Visualize a single frame (e.g., Time Step 4)
# plt.imshow(synthetic_tensor[4, 0, 0].numpy(), cmap='gray')
# plt.title("Synthetic DVS Camera - Time Step 4")
# plt.show()