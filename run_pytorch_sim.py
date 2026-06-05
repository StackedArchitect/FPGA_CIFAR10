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
fc1_w_codes, fc1_wp, fc1_wn, fc1_b, fc1_bns, fc1_bnsh = get_layer_params("fc1")
fc2_w_codes, fc2_wp, fc2_wn, fc2_b, _, _ = get_layer_params("fc2", has_bn=False)

def reconstruct_conv_weight(codes, wp, wn, shape):
    flat = np.select([codes==1, codes==-1], [wp, -wn], 0.0)
    return torch.from_numpy(flat).view(*shape).float()

conv1_w = reconstruct_conv_weight(conv1_w_codes, conv1_wp, conv1_wn, (32, 3, 3, 3))
conv2_w = reconstruct_conv_weight(conv2_w_codes, conv2_wp, conv2_wn, (64, 32, 3, 3))
conv3_w = reconstruct_conv_weight(conv3_w_codes, conv3_wp, conv3_wn, (64, 64, 3, 3))
conv4_w = reconstruct_conv_weight(conv4_w_codes, conv4_wp, conv4_wn, (64, 64, 3, 3))

def reconstruct_fc_weight(w_codes, wp, wn, shape):
    out_f, in_f = shape
    padded_in_f = 20 + in_f + 20
    w_flat = np.select([w_codes==1, w_codes==-1], [wp, -wn], 0.0)
    w_reshaped = w_flat.reshape(out_f, padded_in_f)
    w_data = w_reshaped[:, 20 : 20 + in_f]
    return torch.from_numpy(w_data).float()

fc1_w = reconstruct_fc_weight(fc1_w_codes, fc1_wp, fc1_wn, (256, 64))
fc2_w = reconstruct_fc_weight(fc2_w_codes, fc2_wp, fc2_wn, (10, 256))

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
fc1_th = load_hex_mem(os.path.join(hw_dir, "fc1_act_thresh.mem.bak"))
fc2_th = load_hex_mem(os.path.join(hw_dir, "fc2_act_thresh.mem.bak"))

def apply_fc_with_neuron_threshold(x, w, b, th, scale=None, shift=None):
    out_f, in_f = w.shape
    out = torch.zeros(1, out_f).float()
    for n in range(out_f):
        t = th[n]
        x_pruned = x * (x.abs() >= t).float()
        # Ensure bias is float32
        out[0, n] = F.linear(x_pruned, w[n].unsqueeze(0), torch.from_numpy(b[n:n+1]).float()).item()
        
    if scale is not None:
        out = out * torch.from_numpy(scale).float() + torch.from_numpy(shift).float()
        
    out = F.relu(out)
    return out

def run_forward(mode):
    # Conv1
    x = apply_conv_bn_relu_pool(img_tensor, conv1_w, conv1_b, conv1_bns, conv1_bnsh, has_pool=True)
    
    # Threshold before Conv2
    if mode == 1:
        th = torch.from_numpy(conv1_th).float().view(1, -1, 1, 1)
        x = x * (x.abs() >= th).float()
    elif mode == 2:
        th = 0.005301
        x = x * (x.abs() >= th).float()
        
    # Conv2
    x = apply_conv_bn_relu_pool(x, conv2_w, conv2_b, conv2_bns, conv2_bnsh, has_pool=True)
    
    # Threshold before Conv3
    if mode == 1:
        th = torch.from_numpy(conv2_th).float().view(1, -1, 1, 1)
        x = x * (x.abs() >= th).float()
    elif mode == 2:
        th = 0.006595
        x = x * (x.abs() >= th).float()
        
    # Conv3
    x = apply_conv_bn_relu_pool(x, conv3_w, conv3_b, conv3_bns, conv3_bnsh, has_pool=False)
    
    # Threshold before Conv4
    if mode == 1:
        th = torch.from_numpy(conv3_th).float().view(1, -1, 1, 1)
        x = x * (x.abs() >= th).float()
    elif mode == 2:
        th = 0.006968
        x = x * (x.abs() >= th).float()
        
    # Conv4
    x = apply_conv_bn_relu_pool(x, conv4_w, conv4_b, conv4_bns, conv4_bnsh, has_pool=False)
    
    # GAP
    x = F.avg_pool2d(x, 8).view(1, -1)
    
    # FC1 + BN5 + Relu
    if mode == 0:
        out_fc1 = F.linear(x, fc1_w, torch.from_numpy(fc1_b).float())
        out_fc1 = out_fc1 * torch.from_numpy(fc1_bns).float() + torch.from_numpy(fc1_bnsh).float()
        out_fc1 = F.relu(out_fc1)
    elif mode == 1:
        out_fc1 = apply_fc_with_neuron_threshold(x, fc1_w, fc1_b, fc1_th, fc1_bns, fc1_bnsh)
    elif mode == 2:
        th = np.full(256, 0.005189, dtype=np.float32)
        out_fc1 = apply_fc_with_neuron_threshold(x, fc1_w, fc1_b, th, fc1_bns, fc1_bnsh)
        
    # FC2
    if mode == 0:
        out = F.linear(out_fc1, fc2_w, torch.from_numpy(fc2_b).float())
    elif mode == 1:
        out = apply_fc_with_neuron_threshold(out_fc1, fc2_w, fc2_b, fc2_th)
    elif mode == 2:
        th = np.full(10, 0.003840, dtype=np.float32)
        out = apply_fc_with_neuron_threshold(out_fc1, fc2_w, fc2_b, th)
        
    return out

for mode_name, mode in [("Pure TTQ", 0), ("Per-Channel Thresh", 1), ("Layer-Min Thresh", 2)]:
    out = run_forward(mode)
    logits = out.numpy().flatten()
    pred = logits.argmax()
    print(f"\n=== Mode: {mode_name} ===")
    print(f"Prediction: {pred}")
    print("Logits:")
    for c in range(10):
        print(f"  Class {c}: {logits[c]:.4f} (Q16.16: {int(round(logits[c]*65536))})")
