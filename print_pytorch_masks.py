import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

hw_dir = r"c:\Users\ADMIN\Desktop\FPGA_CIFAR10\cifar10_ttq-bn-threshold\hardware"

SCALE = 65536.0

def load_hex_mem(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        vals = []
        for line in f:
            line = line.strip()
            if not line: continue
            val = int(line, 16)
            if val >= 0x80000000:
                val -= 0x100000000
            vals.append(val / SCALE)
    return np.array(vals, dtype=np.float32)

def load_scalar(filepath):
    with open(filepath, 'r') as f:
        val = int(f.read().strip(), 16)
        if val >= 0x80000000:
            val -= 0x100000000
        return float(val / SCALE)

def load_ternary_weights(filepath):
    with open(filepath, 'r') as f:
        codes = [int(line.strip()) for line in f if line.strip()]
    weights = []
    for c in codes:
        if c == 1:
            weights.append(1.0)
        elif c == 3:
            weights.append(-1.0)
        else:
            weights.append(0.0)
    return np.array(weights, dtype=np.float32)

# Load input image
img_data = load_hex_mem(os.path.join(hw_dir, "data_in.mem"))
img_tensor = torch.from_numpy(img_data).view(1, 3, 32, 32).float()

# Load model parameters
def get_layer_params(name, has_bn=True):
    prefix = os.path.join(hw_dir, name)
    w_codes = load_ternary_weights(prefix + "_w_ternary.mem")
    wp = load_scalar(prefix + "_wp.mem")
    wn = load_scalar(prefix + "_wn.mem")
    bias = load_hex_mem(prefix + "_b.mem")
    
    if has_bn:
        bn_scale = load_hex_mem(prefix + "_bn_scale.mem")
        bn_shift = load_hex_mem(prefix + "_bn_shift.mem")
    else:
        bn_scale, bn_shift = None, None
        
    return w_codes, wp, wn, bias, bn_scale, bn_shift

conv1_w_codes, conv1_wp, conv1_wn, conv1_b, conv1_bns, conv1_bnsh = get_layer_params("conv1")
conv2_w_codes, conv2_wp, conv2_wn, conv2_b, conv2_bns, conv2_bnsh = get_layer_params("conv2")
conv3_w_codes, conv3_wp, conv3_wn, conv3_b, conv3_bns, conv3_bnsh = get_layer_params("conv3")
conv4_w_codes, conv4_wp, conv4_wn, conv4_b, conv4_bns, conv4_bnsh = get_layer_params("conv4")

def reconstruct_conv_weight(codes, wp, wn, shape):
    flat = np.select([codes==1, codes==-1], [wp, -wn], 0.0)
    return torch.from_numpy(flat).view(*shape).float()

conv1_w = reconstruct_conv_weight(conv1_w_codes, conv1_wp, conv1_wn, (32, 3, 3, 3))
conv2_w = reconstruct_conv_weight(conv2_w_codes, conv2_wp, conv2_wn, (64, 32, 3, 3))
conv3_w = reconstruct_conv_weight(conv3_w_codes, conv3_wp, conv3_wn, (64, 64, 3, 3))
conv4_w = reconstruct_conv_weight(conv4_w_codes, conv4_wp, conv4_wn, (64, 64, 3, 3))

def apply_conv_bn_relu_pool(x, w, b, scale, shift, has_pool=True, pool_size=2):
    out = F.conv2d(x, w, torch.from_numpy(b).float(), padding=1)
    if scale is not None:
        out = out * torch.from_numpy(scale).float().view(1, -1, 1, 1) + torch.from_numpy(shift).float().view(1, -1, 1, 1)
    out = F.relu(out)
    if has_pool:
        out = F.max_pool2d(out, pool_size)
    return out

# Read thresholds
conv1_th = load_hex_mem(os.path.join(hw_dir, "conv1_act_thresh.mem.bak"))
conv2_th = load_hex_mem(os.path.join(hw_dir, "conv2_act_thresh.mem.bak"))
conv3_th = load_hex_mem(os.path.join(hw_dir, "conv3_act_thresh.mem.bak"))

# Mode: Per-channel thresholding (which is what hardware should do)
# Conv1
x1 = apply_conv_bn_relu_pool(img_tensor, conv1_w, conv1_b, conv1_bns, conv1_bnsh, has_pool=True)
th1 = torch.from_numpy(conv1_th).float().view(1, -1, 1, 1)
mask1 = (x1.abs() >= th1).float()
print(f"Pool1 Output Mask: total={mask1.numel()}, active={int(mask1.sum().item())} ({100*mask1.sum().item()/mask1.numel():.2f}%)")

# Conv2
x2 = apply_conv_bn_relu_pool(x1 * mask1, conv2_w, conv2_b, conv2_bns, conv2_bnsh, has_pool=True)
th2 = torch.from_numpy(conv2_th).float().view(1, -1, 1, 1)
mask2 = (x2.abs() >= th2).float()
print(f"Pool2 Output Mask: total={mask2.numel()}, active={int(mask2.sum().item())} ({100*mask2.sum().item()/mask2.numel():.2f}%)")

# Conv3
x3 = apply_conv_bn_relu_pool(x2 * mask2, conv3_w, conv3_b, conv3_bns, conv3_bnsh, has_pool=False)
th3 = torch.from_numpy(conv3_th).float().view(1, -1, 1, 1)
mask3 = (x3.abs() >= th3).float()
print(f"Conv3 Output Mask: total={mask3.numel()}, active={int(mask3.sum().item())} ({100*mask3.sum().item()/mask3.numel():.2f}%)")
