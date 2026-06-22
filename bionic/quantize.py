import numpy as np

# 1. We are choosing the 3x3 "Omnidirectional" Laplacian Center-Surround Filter.
# Mimics the receptive fields of biological human retina and 
# account for omnidirectional edge detection (vertical, horizontal, and diagonal)

laplace_weights = np.array([
    [-1.0, -1.0, -1.0],
    [-1.0,  8.0, -1.0],
    [-1.0, -1.0, -1.0]
]) # Sum of all values = 0 (no spike if detecting blank surface with no edges)

print("PyTorch Floating-Point Output:")
print(laplace_weights)

# 2. Hardware Constraints
# We are using 8-bit Signed Integers for our STDP Synapses.
# Maximum possible positive value = 127
MAX_INT8 = 127

# 3. Calculate the Scaling Factor
# Find the largest absolute number in the array to scale it to the hardware limit.
max_float_val = np.max(np.abs(laplace_weights))
scale_factor = MAX_INT8 / max_float_val

# 4. Affine Quantization
# Multiply by the scale factor, round to the nearest whole number, and convert to integer
hardware_weights = np.round(laplace_weights * scale_factor).astype(int)

# 5. Hardware Zero-Sum Correction & Overflow Protection
# Ensures the surrounding negative weights balance the center positive weight.
surround_sum = np.sum(hardware_weights) - hardware_weights[1, 1]

# Check if balancing the center would cause an 8-bit integer overflow (> 127)
while abs(surround_sum) > MAX_INT8:
    # If it overflows, we scale the entire integer array down slightly (e.g., -16 becomes -15)
    hardware_weights = np.where(hardware_weights < 0, hardware_weights + 1, hardware_weights)
    # Recalculate the surround sum after scaling down
    surround_sum = np.sum(hardware_weights) - hardware_weights[1, 1]


# Force the center pixel to be the exact mathematical inverse of the surrounds
hardware_weights[1, 1] = -surround_sum


print(f"\nCalculated Scale Factor: {scale_factor:.3f}")
print("Quantized Output:")
print(hardware_weights)