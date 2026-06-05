import os

hw_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\cifar10_ttq-bn-threshold\hardware"

thresh_configs = [
    ("conv1_act_thresh.mem", 32, "0000015B"),  # active
    ("conv2_act_thresh.mem", 64, "000001B0"),  # active
    ("conv3_act_thresh.mem", 64, "000001C9"),  # active
    ("conv4_act_thresh.mem", 64, "00000000"),
    ("fc1_act_thresh.mem", 256, "00000000"),   # zeroed
    ("fc2_act_thresh.mem", 10, "00000000")     # zeroed
]

for filename, lines, hex_val in thresh_configs:
    filepath = os.path.join(hw_dir, filename)
    with open(filepath, "w") as f:
        f.write("\n".join([hex_val] * lines))
    print(f"Wrote {hex_val} to {filename}")
