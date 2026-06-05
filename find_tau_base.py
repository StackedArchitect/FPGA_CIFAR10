import os
import numpy as np

hw_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\cifar10_ttq-bn-threshold\hardware"

def read_ternary_weights(filepath):
    with open(filepath, 'r') as f:
        codes = [int(line.strip()) for line in f if line.strip()]
    return np.array(codes)

def read_hex_thresholds(filepath):
    filepath_bak = filepath + ".bak"
    target = filepath_bak if os.path.exists(backup_path := filepath_bak) else filepath
    with open(target, 'r') as f:
        vals = []
        for line in f:
            line = line.strip()
            if not line: continue
            val = int(line, 16)
            if val >= 0x80000000:
                val -= 0x100000000
            vals.append(val / 65536.0)
    return np.array(vals)

# Let's inspect conv1
w_codes = read_ternary_weights(os.path.join(hw_dir, "conv1_w_ternary.mem"))
# conv1 weight shape is 32 * 3 * 3 * 3 = 864
w_reshaped = w_codes.reshape(32, 27) # 3 * 3 * 3 = 27 inputs per filter

densities = []
for f in range(32):
    n_nonzero = np.count_nonzero(w_reshaped[f] != 0)
    densities.append(n_nonzero / 27.0)

densities = np.array(densities)
thresholds = read_hex_thresholds(os.path.join(hw_dir, "conv1_act_thresh.mem"))

# Since thresholds = tau_base / clamp(densities, 0.01)
# tau_base = thresholds * clamp(densities, 0.01)
clamped_densities = np.maximum(densities, 0.01)
tau_bases = thresholds * clamped_densities

print("Conv1 filter densities:")
print(densities)
print("\nConv1 thresholds:")
print(thresholds)
print("\nReconstructed tau_bases:")
print(tau_bases)
print(f"\nMean reconstructed tau_base: {tau_bases.mean():.6f}")
