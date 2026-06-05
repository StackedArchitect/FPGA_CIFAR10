import os
import numpy as np

hw_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\cifar10_ttq-bn-threshold\hardware"

def read_ternary_weights(filepath):
    with open(filepath, 'r') as f:
        codes = [int(line.strip()) for line in f if line.strip()]
    return np.array(codes)

# 1. conv2
w_conv2 = read_ternary_weights(os.path.join(hw_dir, "conv2_w_ternary.mem"))
w_conv2 = w_conv2.reshape(64, 32 * 3 * 3) # 64 filters, 288 weights each

# 2. conv3
w_conv3 = read_ternary_weights(os.path.join(hw_dir, "conv3_w_ternary.mem"))
w_conv3 = w_conv3.reshape(64, 64 * 3 * 3) # 64 filters, 576 weights each

# 3. conv4
w_conv4 = read_ternary_weights(os.path.join(hw_dir, "conv4_w_ternary.mem"))
w_conv4 = w_conv4.reshape(64, 64 * 3 * 3) # 64 filters, 576 weights each

# 4. fc1
# fc1 weight shape is 256 * 64
w_fc1 = read_ternary_weights(os.path.join(hw_dir, "fc1_w_ternary.mem"))
# Note: fc1 has padding=20 on each end in the .mem file.
# Let's count entries. 256 * (20 + 64 + 20) = 26624.
# Let's reshape and slice out the padding.
w_fc1 = w_fc1.reshape(256, 104)
w_fc1_data = w_fc1[:, 20:84] # 256 neurons, 64 weights each

# 5. fc2
# fc2 weight shape is 10 * 256
w_fc2 = read_ternary_weights(os.path.join(hw_dir, "fc2_w_ternary.mem"))
# Note: fc2 has padding=20 on each end.
w_fc2 = w_fc2.reshape(10, 296)
w_fc2_data = w_fc2[:, 20:276] # 10 neurons, 256 weights each

tau_base = 0.003

for name, w_tensor in [
    ("Conv2", w_conv2),
    ("Conv3", w_conv3),
    ("Conv4", w_conv4),
    ("FC1", w_fc1_data),
    ("FC2", w_fc2_data)
]:
    # Calculate density per row/filter
    n_nonzero = np.count_nonzero(w_tensor != 0, axis=1)
    total = w_tensor.shape[1]
    densities = n_nonzero / float(total)
    clamped = np.maximum(densities, 0.01)
    taus = tau_base / clamped
    tau_min = taus.min()
    tau_max = taus.max()
    print(f"{name:5s}: weights={total}, density_range=[{densities.min():.4f}, {densities.max():.4f}], tau_min={tau_min:.6f} ({int(round(tau_min*65536))}), tau_max={tau_max:.6f} ({int(round(tau_max*65536))})")
