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
# CIFAR-10 TTQ+BN — Inference / Testing + Weight Export to .mem (FPGA)
#
# This script:
#   1. Loads the trained TTQ model from cifar10_ttq_bn_model.pth
#   2. Evaluates full test set accuracy (10,000 images)
#   3. Per-class accuracy breakdown
#   4. Q16.16 range check
#   5. Single-image inference with logit printout
#   6. Exports all weights to ../weights/ in FPGA-ready .mem format
#      - Ternary codes: +1 → 1 (2'b01), -1 → 3 (2'b11), 0 → 0 (2'b00)
#      - BN parameters: folded scale and shift
#      - Q16.16 fixed-point (32-bit two's complement)
# ===========================================================================

# ---- Architecture parameters (must match training script exactly) ----
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


# ===========================================================================
# TTQ Autograd Function (needed for model definition)
# ===========================================================================
class TTQFunction(torch.autograd.Function):
    """
    Trained Ternary Quantization forward/backward.
    Forward:
        Delta = 0.05 * mean(|W|)
        w_q = Wp * (W > Delta) - Wn * (W < -Delta)
    Backward (STE):
        grad_W: straight-through
        grad_Wp: sum of upstream grads where W > Delta
        grad_Wn: -sum of upstream grads where W < -Delta
    """
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
    """Apply TTQ quantization using the autograd function."""
    return TTQFunction.apply(W, Wp, Wn)


# ===========================================================================
# Model class (must match training script exactly)
# ===========================================================================
class CIFAR10_CNN2D_TTQ_BN(nn.Module):
    """
    TTQ 2D CNN with BatchNorm for CIFAR-10.
    Architecture:
        Conv1(3->32)  -> BN1 -> ReLU -> MaxPool(2×2)   → 16×16×32
        Conv2(32->64) -> BN2 -> ReLU -> MaxPool(2×2)   → 8×8×64
        Conv3(64->64) -> BN3 -> ReLU                    → 8×8×64
        Conv4(64->64) -> BN4 -> ReLU                    → 8×8×64
        GlobalAvgPool(8×8 -> 1×1)                        → 64
        FC1(64->256)  -> BN5 -> ReLU -> Dropout(0.3)    → 256
        FC2(256->10)  -> logits                          → 10
    """
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

        # TTQ mode flag
        self.use_ttq = True

        # TTQ learnable scalars
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
        # Conv1 + BN1 + ReLU + Pool
        w = self._get_weight(self.conv1, self.conv1_wp, self.conv1_wn)
        x = F.conv2d(x, w, self.conv1.bias, padding=1)
        x = self.relu(self.bn1(x))
        x = self.pool(x)

        # Conv2 + BN2 + ReLU + Pool
        w = self._get_weight(self.conv2, self.conv2_wp, self.conv2_wn)
        x = F.conv2d(x, w, self.conv2.bias, padding=1)
        x = self.relu(self.bn2(x))
        x = self.pool2(x)

        # Conv3 + BN3 + ReLU
        w = self._get_weight(self.conv3, self.conv3_wp, self.conv3_wn)
        x = F.conv2d(x, w, self.conv3.bias, padding=1)
        x = self.relu(self.bn3(x))

        # Conv4 + BN4 + ReLU
        w = self._get_weight(self.conv4, self.conv4_wp, self.conv4_wn)
        x = F.conv2d(x, w, self.conv4.bias, padding=1)
        x = self.relu(self.bn4(x))

        # GAP
        x = self.gap(x)
        x = x.view(-1, FC1_IN)

        # FC1 + BN5 + ReLU + Dropout
        w = self._get_weight(self.fc1, self.fc1_wp, self.fc1_wn)
        x = F.linear(x, w, self.fc1.bias)
        x = self.relu(self.bn5(x))
        x = self.dropout(x)

        # FC2
        w = self._get_weight(self.fc2, self.fc2_wp, self.fc2_wn)
        x = F.linear(x, w, self.fc2.bias)
        return x


# ===========================================================================
# Q16.16 export utilities
# ===========================================================================
def to_fixed_point_hex(val):
    """Convert a floating-point value to Q16.16 fixed-point hex string."""
    scaled = int(round(val * FIXED_POINT_SCALE))
    if scaled < 0:
        scaled = (1 << 32) + scaled
    return format(scaled & 0xFFFFFFFF, '08X')


def get_folded_bn_params(bn_layer):
    """Compute folded BN scale and shift: scale = γ/√(σ²+ε), shift = β - μ*scale."""
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta  = bn_layer.bias.detach().cpu().numpy()
    mean  = bn_layer.running_mean.detach().cpu().numpy()
    var   = bn_layer.running_var.detach().cpu().numpy()
    scale = gamma / np.sqrt(var + BN_EPS)
    shift = beta - mean * scale
    return scale, shift


def export_ternary_codes(weight_tensor, out_filename, layer_name):
    """
    Export ternary codes for a conv layer.
    Delta = 0.05 * mean(|W|)
    +1 → 1 (2'b01), -1 → 3 (2'b11), 0 → 0 (2'b00)
    Flatten order: [out_ch][in_ch][kH][kW] (PyTorch default)
    """
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
    
    total = len(flat)
    shape_str = '×'.join(str(s) for s in W.shape)
    print(f"  {os.path.basename(out_filename):<28} {shape_str} = {total} entries  "
          f"(+1:{n_pos} 0:{n_zero} -1:{n_neg})  Δ={delta:.6f}")


def export_fc_ternary_codes(weight_tensor, num_inputs, out_filename, layer_name):
    """
    Export ternary codes for an FC layer with PAD=20 zero-padding per neuron.
    For each neuron: [20 zeros] + [ternary codes for inputs] + [20 zeros]
    """
    W = weight_tensor.detach().cpu()
    delta = TTQ_THRESHOLD * W.abs().max().item()
    num_neurons = W.shape[0]
    
    zero_pad = ['0'] * PAD
    n_pos, n_neg, n_zero_data = 0, 0, 0
    lines = []
    
    for n in range(num_neurons):
        lines.extend(zero_pad)  # Leading PAD
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
                n_zero_data += 1
        lines.extend(zero_pad)  # Trailing PAD
    
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    
    entries_per_neuron = PAD + num_inputs + PAD
    total_data = num_neurons * num_inputs
    print(f"  {os.path.basename(out_filename):<28} {num_neurons}×{entries_per_neuron} = {len(lines)} entries  "
          f"(+1:{n_pos} 0:{n_zero_data} -1:{n_neg})  Δ={delta:.6f}")


def export_biases(bias_tensor, out_filename):
    """Export biases in Q16.16 format."""
    b = bias_tensor.detach().cpu().numpy()
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(to_fixed_point_hex(v) for v in b))
    print(f"  {os.path.basename(out_filename):<28} {len(b)} biases  "
          f"range [{b.min():.6f}, {b.max():.6f}]")


def export_bn_params(bn_layer, layer_name, out_dir):
    """Export folded BN scale and shift in Q16.16 format."""
    scale, shift = get_folded_bn_params(bn_layer)
    scale_file = os.path.join(out_dir, f"{layer_name}_bn_scale.mem")
    shift_file = os.path.join(out_dir, f"{layer_name}_bn_shift.mem")
    with open(scale_file, 'w') as f:
        f.write('\n'.join(to_fixed_point_hex(v) for v in scale))
    with open(shift_file, 'w') as f:
        f.write('\n'.join(to_fixed_point_hex(v) for v in shift))
    print(f"  {os.path.basename(scale_file):<28} {len(scale)} channels  "
          f"range [{scale.min():.5f}, {scale.max():.5f}]")
    print(f"  {os.path.basename(shift_file):<28} {len(shift)} channels  "
          f"range [{shift.min():.5f}, {shift.max():.5f}]")


def export_scalar(scalar_param, out_filename, name):
    """Export a single TTQ scalar (Wp or Wn) in Q16.16 format."""
    val = scalar_param.detach().cpu().item()
    hex_val = to_fixed_point_hex(val)
    with open(out_filename, 'w') as f:
        f.write(hex_val)
    print(f"  {os.path.basename(out_filename):<28} {name}={val:.6f}  hex={hex_val}")


# ===========================================================================
# Main script
# ===========================================================================
def main():
    print("=" * 70)
    print("  CIFAR-10 TTQ+BN — Inference & Weight Export")
    print("=" * 70)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_model.pth")
    WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")

    # ---- Check model file ----
    if not os.path.exists(MODEL_PATH):
        print(f"\n[ERROR] Model file not found: {MODEL_PATH}")
        print(f"        Run train_cifar10_ttq.py first to train the model.")
        sys.exit(1)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")
    if device.type == "cuda":
        print(f"[INFO] GPU:    {torch.cuda.get_device_name(0)}")

    # ---- Load test dataset ----
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                              (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, transform=test_transform, download=True)

    use_gpu = (device.type == "cuda")
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False,
                             num_workers=4 if use_gpu else 0, pin_memory=use_gpu)

    # ---- Load model ----
    print(f"\n[INFO] Loading model from {MODEL_PATH}")
    model = CIFAR10_CNN2D_TTQ_BN().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded successfully ({total_params:,} parameters)")

    # Print TTQ scalar values
    print("\n[INFO] TTQ Wp/Wn values:")
    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
        wp_val = getattr(model, f'{name}_wp').item()
        wn_val = getattr(model, f'{name}_wn').item()
        print(f"  {name}: Wp={wp_val:.6f}, Wn={wn_val:.6f}")

    # ================================================================
    # 1. Full test set evaluation
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Full Test Set Evaluation (10,000 images)")
    print(f"{'='*70}")

    correct, total = 0, 0
    class_correct = [0] * 10
    class_total   = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i].item() == label:
                    class_correct[label] += 1

    overall_acc = 100.0 * correct / total
    print(f"\n  Overall accuracy: {correct}/{total} = {overall_acc:.2f}%")

    # ================================================================
    # 2. Per-class accuracy
    # ================================================================
    print(f"\n  Per-class accuracy:")
    print(f"  {'Class':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"  {'-'*40}")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
        else:
            acc = 0.0
        print(f"  {CIFAR10_CLASSES[i]:<12} {class_correct[i]:>8} {class_total[i]:>8} {acc:>9.2f}%")

    # ================================================================
    # 3. Q16.16 range check
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Q16.16 Range Check")
    print(f"{'='*70}")

    max_logit = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            logits = model(images.to(device))
            max_logit = max(max_logit, logits.abs().max().item())

    q16_max = 32767.9999
    status  = "OK" if max_logit < q16_max else "OVERFLOW"
    print(f"\n  max |logit| observed : {max_logit:.4f}")
    print(f"  Q16.16 safe limit    : {q16_max:.0f}")
    print(f"  Status               : {status}")

    if status == "OVERFLOW":
        print(f"  [WARNING] Logit values exceed Q16.16 safe range!")

    # ================================================================
    # 4. Single-image inference
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Single-Image Inference (test_dataset[0])")
    print(f"{'='*70}")

    test_image, test_label = test_dataset[0]
    with torch.no_grad():
        logits   = model(test_image.unsqueeze(0).to(device))
        pred     = logits.argmax(dim=1).item()
        logits_np = logits.cpu().numpy().flatten()
        q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

    print(f"\n  True label      : {test_label} ({CIFAR10_CLASSES[test_label]})")
    print(f"  Predicted class : {pred} ({CIFAR10_CLASSES[pred]})")
    print(f"  Logits (float)  : {np.array2string(logits_np, precision=4, suppress_small=True)}")
    print(f"  Logits (Q16.16) : {q16_logits}")
    if pred == test_label:
        print("  >>> CORRECT <<<")
    else:
        print(f"  >>> WRONG: expected {test_label}, got {pred} <<<")

    # ================================================================
    # 5. Ternary weight distribution
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  Ternary Weight Distribution")
    print(f"{'='*70}")

    total_weights = 0
    total_pos, total_neg, total_zero = 0, 0, 0

    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
        layer = getattr(model, name)
        W = layer.weight.detach().cpu()
        delta = TTQ_THRESHOLD * W.abs().max().item()
        n_pos = (W > delta).sum().item()
        n_neg = (W < -delta).sum().item()
        n_zero = W.numel() - n_pos - n_neg
        n_total = W.numel()
        total_weights += n_total
        total_pos += n_pos
        total_neg += n_neg
        total_zero += n_zero
        print(f"  {name:6s}: +1={n_pos:>6} ({100*n_pos/n_total:5.1f}%)  "
              f" 0={n_zero:>6} ({100*n_zero/n_total:5.1f}%)  "
              f"-1={n_neg:>6} ({100*n_neg/n_total:5.1f}%)  "
              f"Δ={delta:.6f}  total={n_total}")

    print(f"  {'TOTAL':6s}: +1={total_pos:>6} ({100*total_pos/total_weights:5.1f}%)  "
          f" 0={total_zero:>6} ({100*total_zero/total_weights:5.1f}%)  "
          f"-1={total_neg:>6} ({100*total_neg/total_weights:5.1f}%)  "
          f"total={total_weights}")
    sparsity = 100.0 * total_zero / total_weights
    print(f"  Effective sparsity (zeros): {sparsity:.1f}%")

    # ================================================================
    # 6. Export weights to ../weights/
    # ================================================================
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    model_cpu = model.to('cpu')
    model_cpu.eval()

    print(f"\n{'='*70}")
    print(f"  Exporting weights to {os.path.abspath(WEIGHTS_DIR)}/")
    print(f"  Format: Ternary codes + Q16.16 fixed-point, 32-bit hex")
    print(f"  BN:     separate bn_scale/bn_shift per channel")
    print(f"  TTQ:    Wp/Wn scalars per layer")
    print(f"{'='*70}")

    # ---- Conv1 ----
    print(f"\n  --- Conv1 ({INPUT_CH}→{CONV1_OUT_CH}, 3×3, pad=1) ---")
    export_ternary_codes(model_cpu.conv1.weight,
                         os.path.join(WEIGHTS_DIR, "conv1_w_ternary.mem"), "conv1")
    export_biases(model_cpu.conv1.bias,
                  os.path.join(WEIGHTS_DIR, "conv1_b.mem"))
    export_bn_params(model_cpu.bn1, "conv1", WEIGHTS_DIR)
    export_scalar(model_cpu.conv1_wp,
                  os.path.join(WEIGHTS_DIR, "conv1_wp.mem"), "conv1_wp")
    export_scalar(model_cpu.conv1_wn,
                  os.path.join(WEIGHTS_DIR, "conv1_wn.mem"), "conv1_wn")

    # ---- Conv2 ----
    print(f"\n  --- Conv2 ({CONV2_IN_CH}→{CONV2_OUT_CH}, 3×3, pad=1) ---")
    export_ternary_codes(model_cpu.conv2.weight,
                         os.path.join(WEIGHTS_DIR, "conv2_w_ternary.mem"), "conv2")
    export_biases(model_cpu.conv2.bias,
                  os.path.join(WEIGHTS_DIR, "conv2_b.mem"))
    export_bn_params(model_cpu.bn2, "conv2", WEIGHTS_DIR)
    export_scalar(model_cpu.conv2_wp,
                  os.path.join(WEIGHTS_DIR, "conv2_wp.mem"), "conv2_wp")
    export_scalar(model_cpu.conv2_wn,
                  os.path.join(WEIGHTS_DIR, "conv2_wn.mem"), "conv2_wn")

    # ---- Conv3 ----
    print(f"\n  --- Conv3 ({CONV3_IN_CH}→{CONV3_OUT_CH}, 3×3, pad=1, NO pool) ---")
    export_ternary_codes(model_cpu.conv3.weight,
                         os.path.join(WEIGHTS_DIR, "conv3_w_ternary.mem"), "conv3")
    export_biases(model_cpu.conv3.bias,
                  os.path.join(WEIGHTS_DIR, "conv3_b.mem"))
    export_bn_params(model_cpu.bn3, "conv3", WEIGHTS_DIR)
    export_scalar(model_cpu.conv3_wp,
                  os.path.join(WEIGHTS_DIR, "conv3_wp.mem"), "conv3_wp")
    export_scalar(model_cpu.conv3_wn,
                  os.path.join(WEIGHTS_DIR, "conv3_wn.mem"), "conv3_wn")

    # ---- Conv4 ----
    print(f"\n  --- Conv4 ({CONV4_IN_CH}→{CONV4_OUT_CH}, 3×3, pad=1, NO pool) ---")
    export_ternary_codes(model_cpu.conv4.weight,
                         os.path.join(WEIGHTS_DIR, "conv4_w_ternary.mem"), "conv4")
    export_biases(model_cpu.conv4.bias,
                  os.path.join(WEIGHTS_DIR, "conv4_b.mem"))
    export_bn_params(model_cpu.bn4, "conv4", WEIGHTS_DIR)
    export_scalar(model_cpu.conv4_wp,
                  os.path.join(WEIGHTS_DIR, "conv4_wp.mem"), "conv4_wp")
    export_scalar(model_cpu.conv4_wn,
                  os.path.join(WEIGHTS_DIR, "conv4_wn.mem"), "conv4_wn")

    # ---- FC1 ----
    print(f"\n  --- FC1 ({FC1_IN}→{FC1_OUT}) ---")
    export_fc_ternary_codes(model_cpu.fc1.weight, FC1_IN,
                            os.path.join(WEIGHTS_DIR, "fc1_w_ternary.mem"), "fc1")
    export_biases(model_cpu.fc1.bias,
                  os.path.join(WEIGHTS_DIR, "fc1_b.mem"))
    export_bn_params(model_cpu.bn5, "fc1", WEIGHTS_DIR)
    export_scalar(model_cpu.fc1_wp,
                  os.path.join(WEIGHTS_DIR, "fc1_wp.mem"), "fc1_wp")
    export_scalar(model_cpu.fc1_wn,
                  os.path.join(WEIGHTS_DIR, "fc1_wn.mem"), "fc1_wn")

    # ---- FC2 ----
    print(f"\n  --- FC2 ({FC1_OUT}→{FC2_OUT}, no BN, no dropout) ---")
    export_fc_ternary_codes(model_cpu.fc2.weight, FC1_OUT,
                            os.path.join(WEIGHTS_DIR, "fc2_w_ternary.mem"), "fc2")
    export_biases(model_cpu.fc2.bias,
                  os.path.join(WEIGHTS_DIR, "fc2_b.mem"))
    export_scalar(model_cpu.fc2_wp,
                  os.path.join(WEIGHTS_DIR, "fc2_wp.mem"), "fc2_wp")
    export_scalar(model_cpu.fc2_wn,
                  os.path.join(WEIGHTS_DIR, "fc2_wn.mem"), "fc2_wn")

    # ---- Test input ----
    print(f"\n  --- Test input ---")
    image_np = test_image.numpy()
    hex_pixels = []
    for c in range(INPUT_CH):
        for r in range(INPUT_H):
            for k in range(INPUT_W):
                hex_pixels.append(to_fixed_point_hex(image_np[c, r, k]))

    with open(os.path.join(WEIGHTS_DIR, "data_in.mem"), 'w') as f:
        f.write('\n'.join(hex_pixels))
    print(f"  {'data_in.mem':<28} {len(hex_pixels)} pixels (3×32×32 channel-first)")

    with open(os.path.join(WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
        f.write(format(test_label, '08X'))
    print(f"  {'expected_label.mem':<28} label={test_label} ({CIFAR10_CLASSES[test_label]})")

    # ================================================================
    # 7. Export summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"  Model file       : {MODEL_PATH}")
    print(f"  Test accuracy    : {overall_acc:.2f}%")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Weight format    : Ternary codes + Q16.16 (32-bit hex)")
    print(f"  BN parameters    : separate bn_scale + bn_shift per channel")
    print(f"  TTQ scalars      : Wp + Wn per layer (Q16.16)")
    print(f"  Output directory : {os.path.abspath(WEIGHTS_DIR)}/")
    print(f"  Files generated  :")

    total_size = 0
    for fname in sorted(os.listdir(WEIGHTS_DIR)):
        fpath = os.path.join(WEIGHTS_DIR, fname)
        if os.path.isfile(fpath):
            fsize = os.path.getsize(fpath)
            total_size += fsize
            print(f"    {fname:<28} {fsize:>8,} bytes")

    print(f"  {'':28} {'--------':>8}")
    print(f"  {'TOTAL':<28} {total_size:>8,} bytes")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
