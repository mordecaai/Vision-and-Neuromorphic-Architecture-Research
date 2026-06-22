import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class EdgeDetectorSNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. The 3x3 Kernel (Your STDP Synapses)
        # in_channels=1 (Grayscale DVS), out_channels=1 (One LIF neuron per patch)
        # bias=False (Strict hardware parity)
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        
        # 2. The LIF Neuron
        # 'beta' is the voltage retention rate (the inverse of your PyRTL 'leak')
        # We use a surrogate gradient (ATan) to allow backpropagation through the binary spike
        spike_grad = surrogate.atan()
        self.lif = snn.Leaky(beta=0.9, threshold=1.0, spike_grad=spike_grad, reset_mechanism="zero")

    def forward(self, x):
        # x is the video tensor: [time_steps, batch_size, channels, height, width]
        
        # Initialize the hardware memory bucket (v_mem = 0)
        mem = self.lif.init_leaky() 
        spikes = []

        # Iterate through the temporal dimension (Simulating the PyRTL clock ticks)
        for t in range(x.size(0)):
            # 1. Multiply pixels by weights
            current = self.conv(x[t])
            
            # 2. Add to LIF memory, check threshold, and output spike
            spike, mem = self.lif(current, mem)
            
            # 3. Record the output for the loss function
            spikes.append(spike)

        # Return the final flipbook of output spikes
        return torch.stack(spikes)

# Instantiate the model
model = EdgeDetectorSNN()