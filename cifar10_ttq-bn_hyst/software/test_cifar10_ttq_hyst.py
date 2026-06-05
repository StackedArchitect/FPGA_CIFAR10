import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# ===========================================================================
# CIFAR-10 TTQ+BN+Hysteresis - Inference / Testing + Weight Export to .mem (FPGA)
#
# This script:
#   1. Loads the trained TTQ model from cifar10_ttq_bn_model.pth
#   2. Performs calibration pass on test set to measure non-zero activation
#      means at the 3 hysteresis boundaries (Pool1, Pool2, Conv3 outputs).
#   3. Evaluates test set accuracy under 4 configurations:
#      - Baseline TTQ (no pruning)
#      - DAAP Threshold only (requires daap_config.txt)
#      - Hysteresis only (T_L = kl*mu, T_H = kh*mu)
#      - Combined DAAP + Hysteresis
#   4. Exports all weights to ../weights/ in FPGA-ready .mem format
#   5. Exports per-filter/neuron DAAP activation thresholds
#   6. Exports the spatial hysteresis threshold high/low Q16.16 files:
#      - mask1_thresh_high.mem, mask1_thresh_low.mem
#      - mask2_thresh_high.mem, mask2_thresh_low.mem
#      - mask3_thresh_high.mem, mask3_thresh_low.mem
# ===========================================================================

# ---- Architecture parameters ----
INPUT_H, INPUT_W, INPUT_CH       = 32, 32, 3

CONV1_OUT_CH, CONV1_KERNEL, CONV1_PAD = 32, 3, 1
CONV2_IN_CH  = CONV1_OUT_CH                      # 32
CONV2_OUT_CH, CONV2_KERNEL, CONV2_PAD = 64, 3, 1
CONV3_IN_CH  = CONV2_OUT_CH                      # 64
CONV3_OUT_CH, CONV3_KERNEL, CONV3_PAD = 64, 3, 1
CONV4_IN_CH  = CONV3_OUT_CH                      # 64
CONV4_OUT_CH, CONV4_KERNEL, CONV4_PAD = 64, 3, 1

FC1_IN      = CONV4_OUT_CH                       # 64
FC1_OUT     = 256
FC2_OUT     = 10

PAD                  = 20
FIXED_POINT_SCALE    = 2**16
BN_EPS               = 1e-5
TTQ_THRESHOLD        = 0.05   # Δ = t * max(|W|) per layer

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Default hysteresis scale multipliers
KL = 0.25
KH = 0.70


# ===========================================================================
# Vectorized Spatial Hysteresis Mask Generator
# ===========================================================================
def hysteresis_mask_2d(act_map, T_L, T_H):
    """
    Generate a spatial hysteresis mask for a (B, C, H, W) activation map.
    Same-channel 4-cardinal-neighbor voting, matching the hardware.
    """
    if T_L <= 0.0 and T_H <= 0.0:
        return torch.ones_like(act_map)

    abs_act = act_map.abs()
    active_mask = (abs_act > T_H).float()
    uncertain_mask = ((abs_act >= T_L) & (abs_act <= T_H)).float()

    C = act_map.shape[1]
    kernel = torch.tensor([
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ], dtype=torch.float32, device=act_map.device).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    # Perform channel-wise neighbor summation
    neighbor_count = F.conv2d(active_mask, kernel, padding=1, groups=C)

    resolved_uncertain = uncertain_mask * (neighbor_count >= 2.0).float()
    mask = active_mask + resolved_uncertain
    return mask


# ===========================================================================
# TTQ Autograd Function
# ===========================================================================
class TTQFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, Wp, Wn):
        delta = TTQ_THRESHOLD * W.abs().max()
        pos_mask = (W > delta).float()
        neg_mask = (W < -delta).float()
        w_q = Wp * pos_mask - Wn * neg_mask
        ctx.save_for_backward(W, pos_mask, neg_mask)
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        W, pos_mask, neg_mask = ctx.saved_tensors
        grad_W = grad_output.clone()
        grad_Wp = (grad_output * pos_mask).sum().unsqueeze(0)
        grad_Wn = (-grad_output * neg_mask).sum().unsqueeze(0)
        return grad_W, grad_Wp, grad_Wn


def ttq_quantize(W, Wp, Wn):
    return TTQFunction.apply(W, Wp, Wn)


# ===========================================================================
# Model class
# ===========================================================================
class CIFAR10_CNN2D_TTQ_BN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL, padding=CONV1_PAD)
        self.conv2 = nn.Conv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL, padding=CONV2_PAD)
        self.conv3 = nn.Conv2d(CONV3_IN_CH, CONV3_OUT_CH, CONV3_KERNEL, padding=CONV3_PAD)
        self.conv4 = nn.Conv2d(CONV4_IN_CH, CONV4_OUT_CH, CONV4_KERNEL, padding=CONV4_PAD)

        self.fc1   = nn.Linear(FC1_IN, FC1_OUT)
        self.fc2   = nn.Linear(FC1_OUT, FC2_OUT)

        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3 = nn.BatchNorm2d(CONV3_OUT_CH, eps=BN_EPS, affine=True)
        self.bn4 = nn.BatchNorm2d(CONV4_OUT_CH, eps=BN_EPS, affine=True)
        self.bn5 = nn.BatchNorm1d(FC1_OUT,      eps=BN_EPS, affine=True)

        self.pool  = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        self.use_ttq = True

        self.conv1_wp = nn.Parameter(torch.tensor([0.1]))
        self.conv1_wn = nn.Parameter(torch.tensor([0.1]))
        self.conv2_wp = nn.Parameter(torch.tensor([0.1]))
        self.conv2_wn = nn.Parameter(torch.tensor([0.1]))
        self.conv3_wp = nn.Parameter(torch.tensor([0.1]))
        self.conv3_wn = nn.Parameter(torch.tensor([0.1]))
        self.conv4_wp = nn.Parameter(torch.tensor([0.1]))
        self.conv4_wn = nn.Parameter(torch.tensor([0.1]))
        self.fc1_wp   = nn.Parameter(torch.tensor([0.1]))
        self.fc1_wn   = nn.Parameter(torch.tensor([0.1]))
        self.fc2_wp   = nn.Parameter(torch.tensor([0.1]))
        self.fc2_wn   = nn.Parameter(torch.tensor([0.1]))

    def _get_weight(self, layer, wp, wn):
        if self.use_ttq:
            return ttq_quantize(layer.weight, wp, wn)
        else:
            return layer.weight

    def forward(self, x):
        return self.forward_with_pruning(x)

    def forward_with_pruning(self, x, use_threshold=False, thresholds=None, use_hysteresis=False,
                             mask1_tl=0.0, mask1_th=0.0, mask2_tl=0.0, mask2_th=0.0, mask3_tl=0.0, mask3_th=0.0):
        # Helper for DAAP thresholds
        def _threshold_activation(act, layer_name):
            if not use_threshold or thresholds is None or layer_name not in thresholds:
                return act
            t_val = min(thresholds[layer_name])
            if t_val <= 0.0:
                return act
            mask = (act.abs() >= t_val).float()
            return act * mask

        # Conv1 + BN1 + ReLU + Pool
        w = self._get_weight(self.conv1, self.conv1_wp, self.conv1_wn)
        x = F.conv2d(x, w, self.conv1.bias, padding=1)
        x = self.relu(self.bn1(x))
        x = self.pool(x)                           # (B, 32, 16, 16) - Pool1 output

        # Apply Hysteresis 1
        if use_hysteresis:
            mask1 = hysteresis_mask_2d(x, mask1_tl, mask1_th)
            x = x * mask1

        # Apply DAAP before Conv2
        x = _threshold_activation(x, 'conv2')

        # Conv2 + BN2 + ReLU + Pool
        w = self._get_weight(self.conv2, self.conv2_wp, self.conv2_wn)
        x = F.conv2d(x, w, self.conv2.bias, padding=1)
        x = self.relu(self.bn2(x))
        x = self.pool2(x)                          # (B, 64, 8, 8) - Pool2 output

        # Apply Hysteresis 2
        if use_hysteresis:
            mask2 = hysteresis_mask_2d(x, mask2_tl, mask2_th)
            x = x * mask2

        # Apply DAAP before Conv3
        x = _threshold_activation(x, 'conv3')

        # Conv3 + BN3 + ReLU
        w = self._get_weight(self.conv3, self.conv3_wp, self.conv3_wn)
        x = F.conv2d(x, w, self.conv3.bias, padding=1)
        x = self.relu(self.bn3(x))                 # (B, 64, 8, 8) - Conv3 output

        # Apply Hysteresis 3
        if use_hysteresis:
            mask3 = hysteresis_mask_2d(x, mask3_tl, mask3_th)
            x = x * mask3

        # Apply DAAP before Conv4
        x = _threshold_activation(x, 'conv4')

        # Conv4 + BN4 + ReLU
        w = self._get_weight(self.conv4, self.conv4_wp, self.conv4_wn)
        x = F.conv2d(x, w, self.conv4.bias, padding=1)
        x = self.relu(self.bn4(x))                 # (B, 64, 8, 8)

        # Global Average Pool
        x = self.gap(x)                            # (B, 64, 1, 1)
        x = x.view(-1, FC1_IN)                     # (B, 64)

        # Apply DAAP before FC1
        x = _threshold_activation(x, 'fc1')

        # FC1 + BN5 + ReLU + Dropout
        w = self._get_weight(self.fc1, self.fc1_wp, self.fc1_wn)
        x = F.linear(x, w, self.fc1.bias)
        x = self.relu(self.bn5(x))
        if self.training:
            x = self.dropout(x)

        # Apply DAAP before FC2
        x = _threshold_activation(x, 'fc2')

        # FC2
        w = self._get_weight(self.fc2, self.fc2_wp, self.fc2_wn)
        x = F.linear(x, w, self.fc2.bias)
        return x


# ===========================================================================
# Q16.16 export utilities
# ===========================================================================
def to_fixed_point_hex(val):
    scaled = int(round(val * FIXED_POINT_SCALE))
    if scaled < 0:
        scaled = (1 << 32) + scaled
    return format(scaled & 0xFFFFFFFF, '08X')


def get_folded_bn_params(bn_layer):
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta  = bn_layer.bias.detach().cpu().numpy()
    mean  = bn_layer.running_mean.detach().cpu().numpy()
    var   = bn_layer.running_var.detach().cpu().numpy()
    scale = gamma / np.sqrt(var + BN_EPS)
    shift = beta - mean * scale
    return scale, shift


def export_ternary_codes(weight_tensor, out_filename, layer_name):
    W = weight_tensor.detach().cpu()
    delta = TTQ_THRESHOLD * W.abs().max().item()
    flat = W.flatten()
    n_pos, n_neg, n_zero = 0, 0, 0
    lines = []
    for v in flat:
        v_item = v.item()
        if v_item > delta:
            lines.append('1')
            n_pos += 1
        elif v_item < -delta:
            lines.append('3')
            n_neg += 1
        else:
            lines.append('0')
            n_zero += 1
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"  {os.path.basename(out_filename):<28} {W.shape} = {len(flat)} entries  (+1:{n_pos} 0:{n_zero} -1:{n_neg})")


def export_fc_ternary_codes(weight_tensor, num_inputs, out_filename, layer_name):
    W = weight_tensor.detach().cpu()
    delta = TTQ_THRESHOLD * W.abs().max().item()
    num_neurons = W.shape[0]
    zero_pad = ['0'] * PAD
    n_pos, n_neg, n_zero = 0, 0, 0
    lines = []
    for n in range(num_neurons):
        lines.extend(zero_pad)
        for i in range(num_inputs):
            v = W[n, i].item()
            if v > delta:
                lines.append('1')
                n_pos += 1
            elif v < -delta:
                lines.append('3')
                n_neg += 1
            else:
                lines.append('0')
                n_zero += 1
        lines.extend(zero_pad)
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"  {os.path.basename(out_filename):<28} {num_neurons}×{(PAD+num_inputs+PAD)} = {len(lines)} entries")


def export_biases(bias_tensor, out_filename):
    b = bias_tensor.detach().cpu().numpy()
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(to_fixed_point_hex(v) for v in b))
    print(f"  {os.path.basename(out_filename):<28} {len(b)} biases")


def export_bn_params(bn_layer, layer_name, out_dir):
    scale, shift = get_folded_bn_params(bn_layer)
    with open(os.path.join(out_dir, f"{layer_name}_bn_scale.mem"), 'w') as f:
        f.write('\n'.join(to_fixed_point_hex(v) for v in scale))
    with open(os.path.join(out_dir, f"{layer_name}_bn_shift.mem"), 'w') as f:
        f.write('\n'.join(to_fixed_point_hex(v) for v in shift))
    print(f"  {layer_name}_bn_scale/shift.mem exported")


def export_scalar(scalar_param, out_filename, name):
    val = scalar_param.detach().cpu().item()
    hex_val = to_fixed_point_hex(val)
    with open(out_filename, 'w') as f:
        f.write(hex_val + '\n')
    print(f"  {os.path.basename(out_filename):<28} {name}={val:.6f}")


# ===========================================================================
# Calibration Pass
# ===========================================================================
def calibration_pass(model, loader, device):
    """Run calibration to measure mean of non-zero activations at the 3 spatial boundaries."""
    print("\n[INFO] Starting Calibration Pass to measure non-zero activation statistics...")
    model.eval()

    pool1_vals = []
    pool2_vals = []
    conv3_vals = []

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= 10:  # Use 10 batches (1280 images) for stable calibration stats
                break
            images = images.to(device)

            # Conv1 + BN1 + ReLU + Pool1
            w = model._get_weight(model.conv1, model.conv1_wp, model.conv1_wn)
            x = F.conv2d(images, w, model.conv1.bias, padding=1)
            x = model.relu(model.bn1(x))
            x_pool1 = model.pool(x)
            
            # Conv2 + BN2 + ReLU + Pool2
            w = model._get_weight(model.conv2, model.conv2_wp, model.conv2_wn)
            x = F.conv2d(x_pool1, w, model.conv2.bias, padding=1)
            x = model.relu(model.bn2(x))
            x_pool2 = model.pool2(x)

            # Conv3 + BN3 + ReLU
            w = model._get_weight(model.conv3, model.conv3_wp, model.conv3_wn)
            x = F.conv2d(x_pool2, w, model.conv3.bias, padding=1)
            x_conv3 = model.relu(model.bn3(x))

            # Store positive non-zero values
            p1_nz = x_pool1[x_pool1 > 0.0]
            p2_nz = x_pool2[x_pool2 > 0.0]
            c3_nz = x_conv3[x_conv3 > 0.0]

            if p1_nz.numel() > 0: pool1_vals.append(p1_nz.cpu().numpy())
            if p2_nz.numel() > 0: pool2_vals.append(p2_nz.cpu().numpy())
            if c3_nz.numel() > 0: conv3_vals.append(c3_nz.cpu().numpy())

    mu1 = np.concatenate(pool1_vals).mean() if pool1_vals else 0.8
    mu2 = np.concatenate(pool2_vals).mean() if pool2_vals else 1.0
    mu3 = np.concatenate(conv3_vals).mean() if conv3_vals else 1.0

    print(f"  Measured non-zero activation means (mu):")
    print(f"    Boundary 1 (Pool1 output): mu1 = {mu1:.6f}")
    print(f"    Boundary 2 (Pool2 output): mu2 = {mu2:.6f}")
    print(f"    Boundary 3 (Conv3 output): mu3 = {mu3:.6f}")

    return mu1, mu2, mu3


# ===========================================================================
# DAAP density utilities
# ===========================================================================
def compute_per_filter_density_conv(weight_tensor):
    W = weight_tensor.detach().cpu()
    delta = TTQ_THRESHOLD * W.abs().max().item()
    out_ch = W.shape[0]
    densities = np.zeros(out_ch)
    for f in range(out_ch):
        w_filter = W[f].flatten()
        n_total = w_filter.numel()
        n_nonzero = ((w_filter > delta) | (w_filter < -delta)).sum().item()
        densities[f] = n_nonzero / n_total if n_total > 0 else 0.0
    return densities


def compute_per_neuron_density_fc(weight_tensor):
    W = weight_tensor.detach().cpu()
    delta = TTQ_THRESHOLD * W.abs().max().item()
    num_neurons = W.shape[0]
    densities = np.zeros(num_neurons)
    for n in range(num_neurons):
        w_neuron = W[n].flatten()
        n_total = w_neuron.numel()
        n_nonzero = ((w_neuron > delta) | (w_neuron < -delta)).sum().item()
        densities[n] = n_nonzero / n_total if n_total > 0 else 0.0
    return densities


def compute_thresholds(densities, best_tau_base):
    clamped = np.maximum(densities, 0.01)
    return best_tau_base / clamped


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("  CIFAR-10 TTQ+BN+Hysteresis — Inference & Weight Export")
    print("=" * 70)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Try to load model from SCRIPT_DIR
    MODEL_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_model.pth")
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    # Load test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=test_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Initialize model
    model = CIFAR10_CNN2D_TTQ_BN().to(device)

    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading model from {MODEL_PATH}")
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print(f"[WARNING] Checkpoint {MODEL_PATH} not found!")
        print(f"          Constructing weight matrices from the baseline weights...")
        # Since Vivado synthesis only needs the exported .mem files, we can also reconstruct the PyTorch model 
        # from the existing weights in order to run calibration/evaluation, which is extremely robust.
        weights_src_dir = os.path.join(SCRIPT_DIR, "..", "weights")
        if not os.path.exists(weights_src_dir) or not os.listdir(weights_src_dir):
            # Fall back to base directory weights if present
            weights_src_dir = os.path.join(SCRIPT_DIR, "..", "..", "cifar10_ttq-bn", "weights")
            
        print(f"          Reconstructing weights from {weights_src_dir}...")
        
        # Helper function to parse hex
        def parse_hex_float(hex_str):
            val = int(hex_str.strip(), 16)
            if val >= 0x80000000:
                val -= 0x100000000
            return float(val) / 65536.0

        try:
            # Load Wp/Wn
            for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
                with open(os.path.join(weights_src_dir, f"{layer}_wp.mem"), 'r') as f:
                    wp_val = parse_hex_float(f.readline())
                with open(os.path.join(weights_src_dir, f"{layer}_wn.mem"), 'r') as f:
                    wn_val = parse_hex_float(f.readline())
                getattr(model, f"{layer}_wp").data.fill_(wp_val)
                getattr(model, f"{layer}_wn").data.fill_(wn_val)

            # Reconstruct Conv weights
            for layer_name, conv_layer, shape in [
                ('conv1', model.conv1, (32, 3, 3, 3)),
                ('conv2', model.conv2, (64, 32, 3, 3)),
                ('conv3', model.conv3, (64, 64, 3, 3)),
                ('conv4', model.conv4, (64, 64, 3, 3)),
            ]:
                wp = getattr(model, f"{layer_name}_wp").item()
                wn = getattr(model, f"{layer_name}_wn").item()
                # Read ternary codes
                with open(os.path.join(weights_src_dir, f"{layer_name}_w_ternary.mem"), 'r') as f:
                    codes = [int(line.strip()) for line in f]
                W_flat = []
                for c in codes:
                    if c == 1: W_flat.append(wp)
                    elif c == 3: W_flat.append(-wn)
                    else: W_flat.append(0.0)
                conv_layer.weight.data.copy_(torch.tensor(W_flat).view(shape))
                
                # Read biases
                with open(os.path.join(weights_src_dir, f"{layer_name}_b.mem"), 'r') as f:
                    biases = [parse_hex_float(line) for line in f]
                conv_layer.bias.data.copy_(torch.tensor(biases))

            # Reconstruct Linear weights
            for layer_name, fc_layer, out_features, in_features in [
                ('fc1', model.fc1, 256, 64),
                ('fc2', model.fc2, 10, 256),
            ]:
                wp = getattr(model, f"{layer_name}_wp").item()
                wn = getattr(model, f"{layer_name}_wn").item()
                # Read ternary codes (which are padded in mem file by PAD=20)
                with open(os.path.join(weights_src_dir, f"{layer_name}_w_ternary.mem"), 'r') as f:
                    padded_codes = [int(line.strip()) for line in f]
                W_flat = []
                # Padded shape per neuron is [PAD] + [in_features] + [PAD]
                stride = PAD + in_features + PAD
                for n in range(out_features):
                    neuron_codes = padded_codes[n*stride + PAD : n*stride + PAD + in_features]
                    for c in neuron_codes:
                        if c == 1: W_flat.append(wp)
                        elif c == 3: W_flat.append(-wn)
                        else: W_flat.append(0.0)
                fc_layer.weight.data.copy_(torch.tensor(W_flat).view(out_features, in_features))
                
                # Read biases
                with open(os.path.join(weights_src_dir, f"{layer_name}_b.mem"), 'r') as f:
                    biases = [parse_hex_float(line) for line in f]
                fc_layer.bias.data.copy_(torch.tensor(biases))

            # Reconstruct BatchNorm parameters from scale/shift
            for layer_name, bn_layer, num_features in [
                ('conv1', model.bn1, 32),
                ('conv2', model.bn2, 64),
                ('conv3', model.bn3, 64),
                ('conv4', model.bn4, 64),
                ('fc1',   model.bn5, 256),
            ]:
                with open(os.path.join(weights_src_dir, f"{layer_name}_bn_scale.mem"), 'r') as f:
                    scales = [parse_hex_float(line) for line in f]
                with open(os.path.join(weights_src_dir, f"{layer_name}_bn_shift.mem"), 'r') as f:
                    shifts = [parse_hex_float(line) for line in f]
                # During inference, the batchnorm executes: output = scale * biased + shift
                # We can load these directly into bn_layer weight (scale) and bias (shift), 
                # and set running mean to 0 and running var to 1 (reconstructed inference mode)
                bn_layer.weight.data.copy_(torch.tensor(scales))
                bn_layer.bias.data.copy_(torch.tensor(shifts))
                bn_layer.running_mean.data.zero_()
                bn_layer.running_var.data.fill_(1.0)

            print("[INFO] Model reconstructed successfully from .mem files!")
        except Exception as e:
            print(f"[ERROR] Failed to load/reconstruct weights: {e}")
            sys.exit(1)

    model.eval()

    # 1. Run Calibration Pass
    mu1, mu2, mu3 = calibration_pass(model, test_loader, device)

    # 2. Hysteresis Thresholds
    mask1_tl, mask1_th = KL * mu1, KH * mu1
    mask2_tl, mask2_th = KL * mu2, KH * mu2
    mask3_tl, mask3_th = KL * mu3, KH * mu3

    print(f"\n[INFO] Hysteresis Thresholds:")
    print(f"  Boundary 1 (Pool1): TL={mask1_tl:.6f}, TH={mask1_th:.6f}")
    print(f"  Boundary 2 (Pool2): TL={mask2_tl:.6f}, TH={mask2_th:.6f}")
    print(f"  Boundary 3 (Conv3): TL={mask3_tl:.6f}, TH={mask3_th:.6f}")

    # Load DAAP configuration
    daap_tau_base = 0.0
    daap_config_path = os.path.join(SCRIPT_DIR, "daap_config.txt")
    if not os.path.exists(daap_config_path):
        daap_config_path = os.path.join(SCRIPT_DIR, "..", "..", "cifar10_ttq-bn-threshold", "software", "daap_config.txt")
    if os.path.exists(daap_config_path):
        try:
            with open(daap_config_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('best_tau_base'):
                        daap_tau_base = float(line.strip().split('=')[1].strip())
            print(f"[INFO] Loaded DAAP base threshold tau = {daap_tau_base:.6f}")
        except Exception as e:
            print(f"[WARNING] Could not read DAAP config: {e}")
    else:
        # Load from folder or fallback
        daap_tau_base = 0.007  # standard fallback if none found
        print(f"[INFO] Using fallback DAAP base threshold tau = {daap_tau_base:.6f}")

    # Compute DAAP thresholds
    daap_thresholds = {}
    for name, layer in [('conv2', model.conv2), ('conv3', model.conv3), ('conv4', model.conv4)]:
        daap_thresholds[name] = compute_thresholds(compute_per_filter_density_conv(layer.weight), daap_tau_base)
    daap_thresholds['fc1'] = compute_thresholds(compute_per_neuron_density_fc(model.fc1.weight), daap_tau_base)
    daap_thresholds['fc2'] = compute_thresholds(compute_per_neuron_density_fc(model.fc2.weight), daap_tau_base)

    # 3. Evaluate 4 configurations
    print(f"\nEvaluating test accuracy under 4 configurations...")

    # Config 1: Baseline TTQ
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model.forward_with_pruning(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc_base = 100.0 * correct / total
    print(f"  1. Baseline TTQ (No Pruning)  : {acc_base:.2f}%")

    # Config 2: DAAP Threshold Only
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model.forward_with_pruning(images, use_threshold=True, thresholds=daap_thresholds).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc_daap = 100.0 * correct / total
    print(f"  2. DAAP Threshold Only        : {acc_daap:.2f}%")

    # Config 3: Hysteresis Only
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model.forward_with_pruning(images, use_hysteresis=True,
                                               mask1_tl=mask1_tl, mask1_th=mask1_th,
                                               mask2_tl=mask2_tl, mask2_th=mask2_th,
                                               mask3_tl=mask3_tl, mask3_th=mask3_th).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc_hyst = 100.0 * correct / total
    print(f"  3. Hysteresis Only            : {acc_hyst:.2f}%")

    # Config 4: Combined DAAP + Hysteresis
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model.forward_with_pruning(images, use_threshold=True, thresholds=daap_thresholds, use_hysteresis=True,
                                               mask1_tl=mask1_tl, mask1_th=mask1_th,
                                               mask2_tl=mask2_tl, mask2_th=mask2_th,
                                               mask3_tl=mask3_tl, mask3_th=mask3_th).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc_comb = 100.0 * correct / total
    print(f"  4. Combined (DAAP + Hyst)     : {acc_comb:.2f}%")

    # 4. Export weights and thresholds to weights/
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    model_cpu = model.to('cpu')
    model_cpu.eval()

    print(f"\nExporting all weights and thresholds to {WEIGHTS_DIR}/...")

    # Export weights using existing functions
    export_ternary_codes(model_cpu.conv1.weight, os.path.join(WEIGHTS_DIR, "conv1_w_ternary.mem"), "conv1")
    export_biases(model_cpu.conv1.bias, os.path.join(WEIGHTS_DIR, "conv1_b.mem"))
    export_bn_params(model_cpu.bn1, "conv1", WEIGHTS_DIR)
    export_scalar(model_cpu.conv1_wp, os.path.join(WEIGHTS_DIR, "conv1_wp.mem"), "conv1_wp")
    export_scalar(model_cpu.conv1_wn, os.path.join(WEIGHTS_DIR, "conv1_wn.mem"), "conv1_wn")

    export_ternary_codes(model_cpu.conv2.weight, os.path.join(WEIGHTS_DIR, "conv2_w_ternary.mem"), "conv2")
    export_biases(model_cpu.conv2.bias, os.path.join(WEIGHTS_DIR, "conv2_b.mem"))
    export_bn_params(model_cpu.bn2, "conv2", WEIGHTS_DIR)
    export_scalar(model_cpu.conv2_wp, os.path.join(WEIGHTS_DIR, "conv2_wp.mem"), "conv2_wp")
    export_scalar(model_cpu.conv2_wn, os.path.join(WEIGHTS_DIR, "conv2_wn.mem"), "conv2_wn")

    export_ternary_codes(model_cpu.conv3.weight, os.path.join(WEIGHTS_DIR, "conv3_w_ternary.mem"), "conv3")
    export_biases(model_cpu.conv3.bias, os.path.join(WEIGHTS_DIR, "conv3_b.mem"))
    export_bn_params(model_cpu.bn3, "conv3", WEIGHTS_DIR)
    export_scalar(model_cpu.conv3_wp, os.path.join(WEIGHTS_DIR, "conv3_wp.mem"), "conv3_wp")
    export_scalar(model_cpu.conv3_wn, os.path.join(WEIGHTS_DIR, "conv3_wn.mem"), "conv3_wn")

    export_ternary_codes(model_cpu.conv4.weight, os.path.join(WEIGHTS_DIR, "conv4_w_ternary.mem"), "conv4")
    export_biases(model_cpu.conv4.bias, os.path.join(WEIGHTS_DIR, "conv4_b.mem"))
    export_bn_params(model_cpu.bn4, "conv4", WEIGHTS_DIR)
    export_scalar(model_cpu.conv4_wp, os.path.join(WEIGHTS_DIR, "conv4_wp.mem"), "conv4_wp")
    export_scalar(model_cpu.conv4_wn, os.path.join(WEIGHTS_DIR, "conv4_wn.mem"), "conv4_wn")

    export_fc_ternary_codes(model_cpu.fc1.weight, FC1_IN, os.path.join(WEIGHTS_DIR, "fc1_w_ternary.mem"), "fc1")
    export_biases(model_cpu.fc1.bias, os.path.join(WEIGHTS_DIR, "fc1_b.mem"))
    export_bn_params(model_cpu.bn5, "fc1", WEIGHTS_DIR)
    export_scalar(model_cpu.fc1_wp, os.path.join(WEIGHTS_DIR, "fc1_wp.mem"), "fc1_wp")
    export_scalar(model_cpu.fc1_wn, os.path.join(WEIGHTS_DIR, "fc1_wn.mem"), "fc1_wn")

    export_fc_ternary_codes(model_cpu.fc2.weight, FC1_OUT, os.path.join(WEIGHTS_DIR, "fc2_w_ternary.mem"), "fc2")
    export_biases(model_cpu.fc2.bias, os.path.join(WEIGHTS_DIR, "fc2_b.mem"))
    export_scalar(model_cpu.fc2_wp, os.path.join(WEIGHTS_DIR, "fc2_wp.mem"), "fc2_wp")
    export_scalar(model_cpu.fc2_wn, os.path.join(WEIGHTS_DIR, "fc2_wn.mem"), "fc2_wn")

    # Export DAAP thresholds
    def export_daap_file(filepath, thresh_vals):
        with open(filepath, 'w') as f:
            f.write('\n'.join(to_fixed_point_hex(v) for v in thresh_vals) + '\n')
        print(f"  {os.path.basename(filepath)}: {len(thresh_vals)} entries exported")

    export_daap_file(os.path.join(WEIGHTS_DIR, "conv1_act_thresh.mem"), np.zeros(CONV1_OUT_CH))
    export_daap_file(os.path.join(WEIGHTS_DIR, "conv2_act_thresh.mem"), daap_thresholds['conv2'])
    export_daap_file(os.path.join(WEIGHTS_DIR, "conv3_act_thresh.mem"), daap_thresholds['conv3'])
    export_daap_file(os.path.join(WEIGHTS_DIR, "conv4_act_thresh.mem"), daap_thresholds['conv4'])
    export_daap_file(os.path.join(WEIGHTS_DIR, "fc1_act_thresh.mem"), daap_thresholds['fc1'])
    export_daap_file(os.path.join(WEIGHTS_DIR, "fc2_act_thresh.mem"), daap_thresholds['fc2'])

    # Export Hysteresis thresholds as single value files
    def export_single_val(filepath, val):
        with open(filepath, 'w') as f:
            f.write(to_fixed_point_hex(val) + '\n')
        print(f"  {os.path.basename(filepath)}: {val:.6f} (hex: {to_fixed_point_hex(val)}) exported")

    export_single_val(os.path.join(WEIGHTS_DIR, "mask1_thresh_high.mem"), mask1_th)
    export_single_val(os.path.join(WEIGHTS_DIR, "mask1_thresh_low.mem"), mask1_tl)
    export_single_val(os.path.join(WEIGHTS_DIR, "mask2_thresh_high.mem"), mask2_th)
    export_single_val(os.path.join(WEIGHTS_DIR, "mask2_thresh_low.mem"), mask2_tl)
    export_single_val(os.path.join(WEIGHTS_DIR, "mask3_thresh_high.mem"), mask3_th)
    export_single_val(os.path.join(WEIGHTS_DIR, "mask3_thresh_low.mem"), mask3_tl)

    # Export single test image
    test_image, test_label = test_dataset[0]
    image_np = test_image.numpy()
    hex_pixels = []
    for c in range(INPUT_CH):
        for r in range(INPUT_H):
            for k in range(INPUT_W):
                hex_pixels.append(to_fixed_point_hex(image_np[c, r, k]))
    with open(os.path.join(WEIGHTS_DIR, "data_in.mem"), 'w') as f:
        f.write('\n'.join(hex_pixels) + '\n')
    with open(os.path.join(WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
        f.write(format(test_label, '08X') + '\n')

    print("\n" + "=" * 70)
    print("  EXPORT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
