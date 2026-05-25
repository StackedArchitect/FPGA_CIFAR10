import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# ===========================================================================
# Full-Precision + BatchNorm  2D CNN for MNIST  (FPGA-targeted, Zynq-7020)
#
# Architecture (matches TTQ+BN / TWN+BN for comparison):
#   Conv1 -> BN1 -> ReLU -> Pool1      [weights: full precision float32]
#   Conv2 -> BN2 -> ReLU -> Pool2
#   Flatten
#   FC1   -> BN3 -> ReLU
#   FC2   -> logits                    [no BN on output layer]
#
# This script:
#   1. Trains the BN model
#   2. Exports raw (un-folded) weights + biases as Q16.16 .mem files
#   3. Exports folded BN parameters (bn_scale, bn_shift) per channel
#      for explicit BN hardware states (Approach B)
#
# Folded BN:
#   y = gamma * (x - mean) / sqrt(var + eps) + beta
#     = bn_scale * x + bn_shift
#   where:
#     bn_scale = gamma / sqrt(var + eps)
#     bn_shift = beta  - mean * bn_scale
# ===========================================================================

# ---- Architecture parameters (identical to baseline / TTQ+BN) ----
INPUT_H, INPUT_W, INPUT_CH       = 28, 28, 1
CONV1_OUT_CH, CONV1_KERNEL       = 4, 3
CONV1_OUT_H = INPUT_H - CONV1_KERNEL + 1            # 26
CONV1_OUT_W = INPUT_W - CONV1_KERNEL + 1            # 26
POOL1_SIZE  = 2
POOL1_OUT_H = CONV1_OUT_H // POOL1_SIZE             # 13
POOL1_OUT_W = CONV1_OUT_W // POOL1_SIZE             # 13
CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL = 4, 8, 3
CONV2_OUT_H = POOL1_OUT_H - CONV2_KERNEL + 1        # 11
CONV2_OUT_W = POOL1_OUT_W - CONV2_KERNEL + 1        # 11
POOL2_SIZE  = 2
POOL2_OUT_H = CONV2_OUT_H // POOL2_SIZE             # 5
POOL2_OUT_W = CONV2_OUT_W // POOL2_SIZE             # 5
FLATTEN_SIZE = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH  # 200
FC1_OUT, FC2_OUT                 = 32, 10

PAD                  = 20               # Zero-padding for FC weight .mem files
FIXED_POINT_SCALE    = 2**16            # Q16.16
BN_EPS               = 1e-5


# ===========================================================================
# Model Definition
# ===========================================================================
class MNIST_CNN2D_BN(nn.Module):
    """
    Full-precision 2D CNN with BatchNorm for MNIST.
    Same architecture as TWN+BN and TTQ+BN — only the weight
    representation differs (float32 instead of ternary).
    """
    def __init__(self):
        super().__init__()
        # Standard full-precision Conv and FC layers
        self.conv1 = nn.Conv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL)
        self.conv2 = nn.Conv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL)
        self.fc1   = nn.Linear(FLATTEN_SIZE, FC1_OUT)
        self.fc2   = nn.Linear(FC1_OUT, FC2_OUT)

        # BatchNorm — same configuration as TTQ+BN and TWN+BN
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3 = nn.BatchNorm1d(FC1_OUT,      eps=BN_EPS, affine=True)

        self.pool  = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))   # (B, 4, 26, 26)
        x = self.pool(x)                          # (B, 4, 13, 13)
        x = self.relu(self.bn2(self.conv2(x)))   # (B, 8, 11, 11)
        x = self.pool2(x)                         # (B, 8,  5,  5)
        x = x.view(-1, FLATTEN_SIZE)              # (B, 200)
        x = self.relu(self.bn3(self.fc1(x)))     # (B, 32)
        return self.fc2(x)                        # (B, 10) logits


# ===========================================================================
# Utility functions
# ===========================================================================
def to_fixed_point_hex(value, scale=FIXED_POINT_SCALE):
    """Convert float to Q16.16 32-bit two's complement hex string."""
    fixed = int(round(value * scale))
    if fixed < 0:
        fixed = fixed & 0xFFFFFFFF
    return format(fixed, '08X')


def get_folded_bn_params(bn_layer):
    """
    Fold BatchNorm into scale + shift for FPGA inference.

    Training BN:  y = gamma * (x - mean) / sqrt(var + eps) + beta
    Folded form:  y = scale * x + shift
      scale = gamma / sqrt(var + eps)
      shift = beta  - mean * scale
    """
    gamma = bn_layer.weight.detach().cpu().numpy()
    beta  = bn_layer.bias.detach().cpu().numpy()
    mean  = bn_layer.running_mean.detach().cpu().numpy()
    var   = bn_layer.running_var.detach().cpu().numpy()
    eps   = bn_layer.eps
    scale = gamma / np.sqrt(var + eps)
    shift = beta  - mean * scale
    return scale, shift


# ===========================================================================
# Export functions
# ===========================================================================
def export_conv2d_weights(weight_tensor, out_filename):
    """Export Conv2D weights in flat (out_ch, in_ch, kH, kW) order."""
    w = weight_tensor.cpu().numpy()
    out_ch, in_ch, kH, kW = w.shape
    lines = []
    for f in range(out_ch):
        for c in range(in_ch):
            for r in range(kH):
                for k in range(kW):
                    lines.append(to_fixed_point_hex(w[f, c, r, k]))
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"  {os.path.basename(out_filename):<28} {out_ch}×{in_ch}×{kH}×{kW} = {len(lines)} entries")


def export_biases(bias_tensor, out_filename):
    """Export bias vector as Q16.16."""
    b = bias_tensor.cpu().numpy()
    lines = [to_fixed_point_hex(v) for v in b]
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    print(f"  {os.path.basename(out_filename):<28} {len(lines)} biases")


def export_fc_weights(weight_tensor, num_inputs, out_filename):
    """FC weights with PAD zero-padding on each side (same format as baseline)."""
    w = weight_tensor.cpu().numpy()
    num_neurons = w.shape[0]
    padding = ["00000000"] * PAD
    lines = []
    for n in range(num_neurons):
        hex_w = [to_fixed_point_hex(v) for v in w[n]]
        lines.extend(padding + hex_w + padding)
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    entries_per_neuron = PAD + num_inputs + PAD
    print(f"  {os.path.basename(out_filename):<28} {num_neurons}×{entries_per_neuron} = {len(lines)} entries")


def export_bn_params(bn_layer, layer_name, out_dir):
    """Export folded BN scale and shift as Q16.16 per channel."""
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


# ===========================================================================
# Training setup
# ===========================================================================
print("=" * 60)
print("  Full-Precision + BN 2D CNN -- MNIST (Baseline + BN)")
print("=" * 60)
print(f"  Weights      : full precision float32")
print(f"  BatchNorm    : after Conv1, Conv2, FC1")
print(f"  Biases       : full precision")
print(f"  Layers       : conv1, conv2, fc1, fc2")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True,  transform=transform, download=True)
test_dataset  = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transform, download=True)
train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Device: {device}")

model     = MNIST_CNN2D_BN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)


# ===========================================================================
# Training loop
# ===========================================================================
EPOCHS = 15

print(f"\n[INFO] Training for {EPOCHS} epochs")
print(f"{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 44)

best_acc = 0.0
train_start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item()
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    scheduler.step()

    # ---- Evaluation ----
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            preds          = outputs.argmax(dim=1)
            test_correct  += (preds == labels).sum().item()
            test_total    += labels.size(0)

    train_acc = 100 * total_correct / total_samples
    test_acc  = 100 * test_correct  / test_total
    avg_loss  = total_loss / len(train_loader)

    print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% "
          f"{test_acc:>9.2f}%", end="")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "cnn2d_bn_mnist_model.pth")
        print("  <- best saved", end="")
    print()

train_time = time.time() - train_start

print(f"\n[INFO] Training completed in {train_time:.1f}s")
print(f"[INFO] Best test accuracy: {best_acc:.2f}%")


# ===========================================================================
# Load best model and verify
# ===========================================================================
model.load_state_dict(torch.load("cnn2d_bn_mnist_model.pth", map_location=device,
                                  weights_only=True))
model.eval()

# Q16.16 range check
max_logit = 0.0
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        if i >= 10:
            break
        logits    = model(images.to(device))
        max_logit = max(max_logit, logits.abs().max().item())

q16_max = 32767.9999
status  = "OK" if max_logit < q16_max else "OVERFLOW"
print(f"\n  Q16.16 range check:")
print(f"    max |logit| observed : {max_logit:.4f}")
print(f"    Q16.16 safe limit    : {q16_max:.0f}")
print(f"    Status               : {status}")


# Single-image verification
test_image, test_label = test_dataset[0]
with torch.no_grad():
    logits   = model(test_image.unsqueeze(0).to(device))
    pred     = logits.argmax(dim=1).item()
    logits_np = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

print(f"\n  Single-image test (index 0):")
print(f"    True label      : {test_label}")
print(f"    Predicted digit : {pred}")
print(f"    Logits          : {np.array2string(logits_np, precision=4, suppress_small=True)}")
print(f"    Q16.16          : {q16_logits}")
if pred == test_label:
    print("    >>> CORRECT <<<")
else:
    print(f"    >>> WRONG: expected {test_label}, got {pred} <<<")


# ===========================================================================
# Export weights to ../weights_bn/
#
# Approach B (explicit BN hardware):
#   - Raw weights and biases (NOT folded with BN)
#   - Separate BN scale/shift per channel
# ===========================================================================
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "weights_bn")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

model_cpu = model.to('cpu')
model_cpu.eval()   # CRITICAL: running stats must be frozen before export
state_dict = model_cpu.state_dict()

print(f"\n{'='*60}")
print(f"  Exporting weights to {WEIGHTS_DIR}/")
print(f"  Format: Q16.16 fixed-point, 32-bit hex")
print(f"  BN:     separate bn_scale/bn_shift per channel")
print(f"{'='*60}")

# ---- Conv1 ----
print(f"\n  --- Conv1 ---")
export_conv2d_weights(state_dict['conv1.weight'],
                      os.path.join(WEIGHTS_DIR, "conv1_w.mem"))
export_biases(state_dict['conv1.bias'],
              os.path.join(WEIGHTS_DIR, "conv1_b.mem"))
export_bn_params(model_cpu.bn1, "conv1", WEIGHTS_DIR)

# ---- Conv2 ----
print(f"\n  --- Conv2 ---")
export_conv2d_weights(state_dict['conv2.weight'],
                      os.path.join(WEIGHTS_DIR, "conv2_w.mem"))
export_biases(state_dict['conv2.bias'],
              os.path.join(WEIGHTS_DIR, "conv2_b.mem"))
export_bn_params(model_cpu.bn2, "conv2", WEIGHTS_DIR)

# ---- FC1 ----
print(f"\n  --- FC1 ---")
export_fc_weights(state_dict['fc1.weight'], FLATTEN_SIZE,
                  os.path.join(WEIGHTS_DIR, "fc1_w.mem"))
export_biases(state_dict['fc1.bias'],
              os.path.join(WEIGHTS_DIR, "fc1_b.mem"))
export_bn_params(model_cpu.bn3, "fc1", WEIGHTS_DIR)

# ---- FC2 (no BN) ----
print(f"\n  --- FC2 (no BN) ---")
export_fc_weights(state_dict['fc2.weight'], FC1_OUT,
                  os.path.join(WEIGHTS_DIR, "fc2_w.mem"))
export_biases(state_dict['fc2.bias'],
              os.path.join(WEIGHTS_DIR, "fc2_b.mem"))


# ---- Test input ----
print(f"\n  --- Test input ---")
image_np = test_image.squeeze().numpy()
hex_pixels = [to_fixed_point_hex(image_np[r, c])
              for r in range(INPUT_H) for c in range(INPUT_W)]
with open(os.path.join(WEIGHTS_DIR, "data_in.mem"), 'w') as f:
    f.write('\n'.join(hex_pixels))
print(f"  {'data_in.mem':<28} {len(hex_pixels)} pixels (28×28 row-major)")

with open(os.path.join(WEIGHTS_DIR, "expected_label.mem"), 'w') as f:
    f.write(format(test_label, '08X'))
print(f"  {'expected_label.mem':<28} label={test_label}")


# ===========================================================================
# Summary
# ===========================================================================
total_params = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)
print(f"\n{'='*60}")
print(f"  EXPORT COMPLETE")
print(f"{'='*60}")
print(f"  Model accuracy     : {best_acc:.2f}%")
print(f"  Total parameters   : {total_params:,}")
print(f"  Weight format      : Q16.16 (32-bit hex)")
print(f"  BN parameters      : separate bn_scale + bn_shift per channel")
print(f"  Output directory   : {WEIGHTS_DIR}/")
print(f"  Files generated    :")
for fname in sorted(os.listdir(WEIGHTS_DIR)):
    fpath = os.path.join(WEIGHTS_DIR, fname)
    print(f"    {fname:<28} {os.path.getsize(fpath):>8,} bytes")
print(f"{'='*60}")
