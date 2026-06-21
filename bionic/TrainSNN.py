import torch
import torch.optim as optim
import torch.nn.functional as F
from EdgeDetectorSNN import EdgeDetectorSNN
from TensorGenerator import synthetic_tensor

# 1. Setup the Model and Optimizer
model = EdgeDetectorSNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# We want the network to fire when it sees the edge. 
# So our "Target" is a tensor of 1s (asking the LIF neuron to spike).
target_spikes = torch.ones((10, 1, 1, 10, 10)) # 10 time steps, 10x10 LIF output

print("Starting Offline Training...")

# 2. The Training Loop (Run it 50 times)
for epoch in range(500):
    optimizer.zero_grad()
    
    # Run the synthetic video through the bionic eye
    output_spikes = model(synthetic_tensor)
    
    # Calculate how far off the bionic eye was from our target
    # (Using Mean Squared Error for simplicity here)
    loss = F.mse_loss(output_spikes, target_spikes)
    
    # Backpropagation (The Magic)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# 3. Extract the final baked "Lego Bricks"
trained_weights = model.conv.weight.detach().numpy()
print("\nFinal Offline Trained 3x3 Kernel:")
print(trained_weights)