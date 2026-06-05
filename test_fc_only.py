import os

hw_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\cifar10_ttq-bn-threshold\hardware"

thresh_configs = [
    ("conv1_act_thresh.mem", 32, "00000000"),  # zeroed
    ("conv2_act_thresh.mem", 64, "00000000"),  # zeroed
    ("conv3_act_thresh.mem", 64, "00000000"),  # zeroed
    ("conv4_act_thresh.mem", 64, "00000000"),
    ("fc1_act_thresh.mem", 256, "00000154"),   # active
    ("fc2_act_thresh.mem", 10, "000000FC")     # active
]

for filename, lines, hex_val in thresh_configs:
    filepath = os.path.join(hw_dir, filename)
    with open(filepath, "w") as f:
        f.write("\n".join([hex_val] * lines))
    print(f"Wrote {hex_val} to {filename}")
