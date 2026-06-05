import os

hw_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\cifar10_ttq-bn-threshold\hardware"

thresh_configs = [
    ("conv1_act_thresh.mem", 32, "0000015B"),  # tau_min_conv2 = 347 (0x15B)
    ("conv2_act_thresh.mem", 64, "000001B0"),  # tau_min_conv3 = 432 (0x1B0)
    ("conv3_act_thresh.mem", 64, "000001C9"),  # tau_min_conv4 = 457 (0x1C9)
    ("conv4_act_thresh.mem", 64, "00000000"),  # Conv4 output to GAP (no thresholding in PyTorch, set to 0)
    ("fc1_act_thresh.mem", 256, "00000154"),   # tau_min_fc1 = 340 (0x154)
    ("fc2_act_thresh.mem", 10, "000000FC")     # tau_min_fc2 = 252 (0x0FC)
]

for filename, lines, hex_val in thresh_configs:
    filepath = os.path.join(hw_dir, filename)
    with open(filepath, "w") as f:
        f.write("\n".join([hex_val] * lines))
    print(f"Wrote {hex_val} ({lines} lines) to {filename}")
