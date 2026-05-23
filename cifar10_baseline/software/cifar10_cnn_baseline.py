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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Cutout:
    """Randomly mask a square patch of the image (after ToTensor/Normalize)."""
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = torch.ones(h, w, dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        return img * mask

# ===========================================================================
# Full-Precision + BatchNorm  2D CNN for CIFAR-10  (FPGA-targeted, Zynq-7020)
#
# Architecture (v4 — 4 conv layers + KD):
#   Conv1 -> BN1 -> ReLU -> Pool1      [3->32, 3×3, pad=1, pool 2×2]
#   Conv2 -> BN2 -> ReLU -> Pool2      [32->64, 3×3, pad=1, pool 2×2]
#   Conv3 -> BN3 -> ReLU               [64->64, 3×3, pad=1, NO pool]
#   Conv4 -> BN4 -> ReLU               [64->64, 3×3, pad=1, NO pool]
#   Global Average Pool (8×8 -> 1×1)
#   FC1   -> BN5 -> ReLU -> Dropout    [64->256]
#   FC2   -> logits                    [256->10, no BN, no dropout]
#
# Training: Knowledge Distillation from ResNet-18 teacher (~95%)
#   L = α * CE(student, labels) + (1-α) * T² * KL(student_soft, teacher_soft)
#
# The 4th conv layer gives a 5×5 effective receptive field at the 8×8
# spatial resolution — critical for capturing mid-level features.
# This breaks the ~87% ceiling of 3-conv architectures.
#
# FPGA resource budget (XC7Z020, 140 BRAM36):
#   Conv1 weights:   864    LUT-ROM (distributed)
#   Conv2 weights:  18,432  18 BRAM36
#   Conv3 weights:  36,864  36 BRAM36
#   Conv4 weights:  36,864  36 BRAM36
#   FC1 weights:    26,624  26 BRAM36
#   FC2 weights:     2,960   3 BRAM36
#   Buffers:                14 BRAM36
#   TOTAL:                 133 / 140 BRAM36  ✓
#
# Fixed-point: Q16.16 (32-bit, two's complement)
# ===========================================================================

# ---- Architecture parameters ----
INPUT_H, INPUT_W, INPUT_CH       = 32, 32, 3

CONV1_OUT_CH, CONV1_KERNEL, CONV1_PAD = 32, 3, 1
CONV1_OUT_H = INPUT_H                            # 32
CONV1_OUT_W = INPUT_W                            # 32
POOL1_SIZE  = 2
POOL1_OUT_H = CONV1_OUT_H // POOL1_SIZE          # 16
POOL1_OUT_W = CONV1_OUT_W // POOL1_SIZE          # 16

CONV2_IN_CH  = CONV1_OUT_CH                      # 32
CONV2_OUT_CH, CONV2_KERNEL, CONV2_PAD = 64, 3, 1
CONV2_OUT_H = POOL1_OUT_H                        # 16
CONV2_OUT_W = POOL1_OUT_W                        # 16
POOL2_SIZE  = 2
POOL2_OUT_H = CONV2_OUT_H // POOL2_SIZE          # 8
POOL2_OUT_W = CONV2_OUT_W // POOL2_SIZE          # 8

CONV3_IN_CH  = CONV2_OUT_CH                      # 64
CONV3_OUT_CH, CONV3_KERNEL, CONV3_PAD = 64, 3, 1
CONV3_OUT_H = POOL2_OUT_H                        # 8
CONV3_OUT_W = POOL2_OUT_W                        # 8

CONV4_IN_CH  = CONV3_OUT_CH                      # 64
CONV4_OUT_CH, CONV4_KERNEL, CONV4_PAD = 64, 3, 1
CONV4_OUT_H = CONV3_OUT_H                        # 8
CONV4_OUT_W = CONV3_OUT_W                        # 8

GAP_SIZE    = CONV4_OUT_H                        # 8

FC1_IN      = CONV4_OUT_CH                       # 64
FC1_OUT     = 256
FC2_OUT     = 10

PAD                  = 20
FIXED_POINT_SCALE    = 2**16
BN_EPS               = 1e-5

# ---- KD Hyperparameters ----
KD_TEMPERATURE = 4.0
KD_ALPHA       = 0.3


# ===========================================================================
# Student Model — 4-layer CNN with BatchNorm
# ===========================================================================
class CIFAR10_CNN2D_BN(nn.Module):
    """
    Full-precision 2D CNN with BatchNorm for CIFAR-10 (student, v4).
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
        # Convolutional layers
        self.conv1 = nn.Conv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL, padding=CONV1_PAD)
        self.conv2 = nn.Conv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL, padding=CONV2_PAD)
        self.conv3 = nn.Conv2d(CONV3_IN_CH, CONV3_OUT_CH, CONV3_KERNEL, padding=CONV3_PAD)
        self.conv4 = nn.Conv2d(CONV4_IN_CH, CONV4_OUT_CH, CONV4_KERNEL, padding=CONV4_PAD)

        # Fully connected layers
        self.fc1   = nn.Linear(FC1_IN, FC1_OUT)
        self.fc2   = nn.Linear(FC1_OUT, FC2_OUT)

        # BatchNorm — after each conv and after FC1
        self.bn1 = nn.BatchNorm2d(CONV1_OUT_CH, eps=BN_EPS, affine=True)
        self.bn2 = nn.BatchNorm2d(CONV2_OUT_CH, eps=BN_EPS, affine=True)
        self.bn3 = nn.BatchNorm2d(CONV3_OUT_CH, eps=BN_EPS, affine=True)
        self.bn4 = nn.BatchNorm2d(CONV4_OUT_CH, eps=BN_EPS, affine=True)
        self.bn5 = nn.BatchNorm1d(FC1_OUT,      eps=BN_EPS, affine=True)  # FC1 BN

        # Pooling and activation
        self.pool  = nn.MaxPool2d(POOL1_SIZE)
        self.pool2 = nn.MaxPool2d(POOL2_SIZE)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.relu  = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

        # He (Kaiming) initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))    # (B, 32, 32, 32)
        x = self.pool(x)                           # (B, 32, 16, 16)
        x = self.relu(self.bn2(self.conv2(x)))    # (B, 64, 16, 16)
        x = self.pool2(x)                          # (B, 64,  8,  8)
        x = self.relu(self.bn3(self.conv3(x)))    # (B, 64,  8,  8)
        x = self.relu(self.bn4(self.conv4(x)))    # (B, 64,  8,  8) ← NEW
        x = self.gap(x)                            # (B, 64,  1,  1)
        x = x.view(-1, FC1_IN)                     # (B, 64)
        x = self.relu(self.bn5(self.fc1(x)))      # (B, 256)
        x = self.dropout(x)
        return self.fc2(x)                         # (B, 10)


# ===========================================================================
# Teacher Model — ResNet-18 adapted for CIFAR-10
# ===========================================================================
def create_teacher():
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ===========================================================================
# KD Loss
# ===========================================================================
def kd_loss(student_logits, teacher_logits, labels, T=KD_TEMPERATURE, alpha=KD_ALPHA):
    """Knowledge Distillation loss (Hinton et al., 2015)."""
    ce = F.cross_entropy(student_logits, labels)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1).detach()
    kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    return alpha * ce + (1.0 - alpha) * T * T * kl


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


def export_conv2d_weights(weight_tensor, out_filename):
    w = weight_tensor.detach().cpu().numpy()
    flat = w.flatten()
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(to_fixed_point_hex(v) for v in flat))
    shape_str = '×'.join(str(s) for s in w.shape)
    print(f"  {os.path.basename(out_filename):<28} {shape_str} = {len(flat)} entries")


def export_biases(bias_tensor, out_filename):
    b = bias_tensor.detach().cpu().numpy()
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(to_fixed_point_hex(v) for v in b))
    print(f"  {os.path.basename(out_filename):<28} {len(b)} biases")


def export_fc_weights(weight_tensor, num_inputs, out_filename):
    w = weight_tensor.detach().cpu().numpy()
    num_neurons = w.shape[0]
    padding = [to_fixed_point_hex(0.0)] * PAD
    lines = []
    for n in range(num_neurons):
        hex_w = [to_fixed_point_hex(v) for v in w[n]]
        lines.extend(padding + hex_w + padding)
    with open(out_filename, 'w') as fp:
        fp.write('\n'.join(lines))
    entries_per_neuron = PAD + num_inputs + PAD
    print(f"  {os.path.basename(out_filename):<28} {num_neurons}×{entries_per_neuron} = {len(lines)} entries")


def export_bn_params(bn_layer, layer_name, out_dir):
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
# Setup
# ===========================================================================
print("=" * 70)
print("  CIFAR-10 CNN — 4-Layer + Knowledge Distillation (v4)")
print("=" * 70)
print(f"  Teacher      : ResNet-18 (~11M params, ~95%)")
print(f"  Student      : Conv1(3→32)→Conv2(32→64)→Conv3(64→64)→Conv4(64→64)")
print(f"                 →GAP→FC1(64→256)→FC2(256→10)")
print(f"  KD           : T={KD_TEMPERATURE}, α={KD_ALPHA}")
print(f"  Target       : >90% student accuracy")
print("=" * 70)

# ---- Data augmentation ----
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2470, 0.2435, 0.2616)),
    Cutout(n_holes=1, length=16),   # Regularize: mask 16×16 patch (25% of image)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2470, 0.2435, 0.2616))
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True,  transform=train_transform, download=True)
test_dataset  = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transform,  download=True)

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Device: {device}")
if device.type == "cuda":
    print(f"[INFO] GPU:    {torch.cuda.get_device_name(0)}")
    print(f"[INFO] VRAM:   {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True

use_gpu = (device.type == "cuda")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                          num_workers=4 if use_gpu else 0, pin_memory=use_gpu)
test_loader  = DataLoader(test_dataset,  batch_size=256, shuffle=False,
                          num_workers=4 if use_gpu else 0, pin_memory=use_gpu)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEACHER_PATH = os.path.join(SCRIPT_DIR, "resnet18_teacher_cifar10.pth")


def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ===========================================================================
#  PHASE 1: Teacher (skip if cached)
# ===========================================================================
TEACHER_EPOCHS = 200

if os.path.exists(TEACHER_PATH):
    print(f"\n[PHASE 1] Teacher found — loading {TEACHER_PATH}")
    teacher = create_teacher().to(device)
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device, weights_only=True))
    teacher_acc = evaluate(teacher)
    print(f"          Teacher accuracy: {teacher_acc:.2f}%")
else:
    print(f"\n[PHASE 1] Training ResNet-18 teacher ({TEACHER_EPOCHS} epochs)")
    print(f"          SGD(lr=0.1, momentum=0.9, nesterov), MultiStepLR([100,150])")
    print("-" * 60)

    teacher = create_teacher().to(device)
    t_criterion = nn.CrossEntropyLoss()
    t_optimizer = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
    t_scheduler = optim.lr_scheduler.MultiStepLR(t_optimizer,
                                                  milestones=[100, 150], gamma=0.1)
    teacher_best_acc = 0.0
    t_start = time.time()

    for epoch in range(TEACHER_EPOCHS):
        teacher.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            t_optimizer.zero_grad()
            loss = t_criterion(teacher(images), labels)
            loss.backward()
            t_optimizer.step()
        t_scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == TEACHER_EPOCHS - 1:
            acc = evaluate(teacher)
            print(f"  Teacher epoch {epoch+1:>3}/{TEACHER_EPOCHS}  |  "
                  f"test acc: {acc:.2f}%", end="")
            if acc > teacher_best_acc:
                teacher_best_acc = acc
                torch.save(teacher.state_dict(), TEACHER_PATH)
                print("  <- best", end="")
            print()

    t_time = time.time() - t_start
    print(f"\n[PHASE 1] Teacher done in {t_time:.1f}s  (best: {teacher_best_acc:.2f}%)")
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device, weights_only=True))

teacher.eval()


# ===========================================================================
#  PHASE 2: Student with KD (4-layer CNN)
# ===========================================================================
STUDENT_EPOCHS = 350

print(f"\n{'='*70}")
print(f"  [PHASE 2] Training 4-Layer Student with KD ({STUDENT_EPOCHS} epochs)")
print(f"            Adam(lr=1e-3, wd=5e-4), CosineAnnealingLR(T_max={STUDENT_EPOCHS})")
print(f"{'='*70}")
print(f"{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 44)

student = CIFAR10_CNN2D_BN().to(device)
total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f"[INFO] Student parameters: {total_params:,}")

s_optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=5e-4)
s_scheduler = optim.lr_scheduler.CosineAnnealingLR(s_optimizer, T_max=STUDENT_EPOCHS)

best_acc    = 0.0
s_start     = time.time()

history_loss = []
history_train_acc = []
history_test_acc  = []

for epoch in range(STUDENT_EPOCHS):
    student.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)

        s_optimizer.zero_grad()
        student_logits = student(images)
        loss = kd_loss(student_logits, teacher_logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        s_optimizer.step()

        total_loss    += loss.item()
        preds          = student_logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    s_scheduler.step()

    train_acc = 100 * total_correct / total_samples
    test_acc  = evaluate(student)
    avg_loss  = total_loss / len(train_loader)

    history_loss.append(avg_loss)
    history_train_acc.append(train_acc)
    history_test_acc.append(test_acc)

    print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% "
          f"{test_acc:>9.2f}%", end="")
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(student.state_dict(), "cifar10_cnn_baseline_model.pth")
        print("  <- best saved", end="")
    print()

s_time = time.time() - s_start

print(f"\n[PHASE 2] Student training done in {s_time:.1f}s")
print(f"          Best student accuracy: {best_acc:.2f}%")


# ===========================================================================
# Training curve
# ===========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, STUDENT_EPOCHS + 1), history_loss, 'b-', linewidth=1.0)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss (KD)')
ax1.set_title('CIFAR-10 4-Layer Student (KD) — Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, STUDENT_EPOCHS + 1), history_train_acc, 'b-', linewidth=1.0,
         label='Train Acc')
ax2.plot(range(1, STUDENT_EPOCHS + 1), history_test_acc,  'r-', linewidth=1.0,
         label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('CIFAR-10 4-Layer Student (KD) — Accuracy')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "cifar10_training_curve.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[INFO] Training curve saved to {plot_path}")


# ===========================================================================
# Load best model and verify
# ===========================================================================
model = student
model.load_state_dict(torch.load("cifar10_cnn_baseline_model.pth",
                                  map_location=device, weights_only=True))
model.eval()

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

test_image, test_label = test_dataset[0]
with torch.no_grad():
    logits   = model(test_image.unsqueeze(0).to(device))
    pred     = logits.argmax(dim=1).item()
    logits_np = logits.cpu().numpy().flatten()
    q16_logits = (logits_np * FIXED_POINT_SCALE).astype(np.int64)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\n  Single-image test (index 0):")
print(f"    True label      : {test_label} ({CIFAR10_CLASSES[test_label]})")
print(f"    Predicted class : {pred} ({CIFAR10_CLASSES[pred]})")
print(f"    Logits          : {np.array2string(logits_np, precision=4, suppress_small=True)}")
print(f"    Q16.16          : {q16_logits}")
if pred == test_label:
    print("    >>> CORRECT <<<")
else:
    print(f"    >>> WRONG: expected {test_label}, got {pred} <<<")


# ===========================================================================
# Export weights to ../weights/
# ===========================================================================
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, "..", "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

model_cpu = model.to('cpu')
model_cpu.eval()
state_dict = model_cpu.state_dict()

print(f"\n{'='*65}")
print(f"  Exporting weights to {WEIGHTS_DIR}/")
print(f"  Format: Q16.16 fixed-point, 32-bit hex")
print(f"  BN:     separate bn_scale/bn_shift per channel")
print(f"{'='*65}")

# ---- Conv1 ----
print(f"\n  --- Conv1 ({INPUT_CH}→{CONV1_OUT_CH}, 3×3, pad=1) ---")
export_conv2d_weights(state_dict['conv1.weight'],
                      os.path.join(WEIGHTS_DIR, "conv1_w.mem"))
export_biases(state_dict['conv1.bias'],
              os.path.join(WEIGHTS_DIR, "conv1_b.mem"))
export_bn_params(model_cpu.bn1, "conv1", WEIGHTS_DIR)

# ---- Conv2 ----
print(f"\n  --- Conv2 ({CONV2_IN_CH}→{CONV2_OUT_CH}, 3×3, pad=1) ---")
export_conv2d_weights(state_dict['conv2.weight'],
                      os.path.join(WEIGHTS_DIR, "conv2_w.mem"))
export_biases(state_dict['conv2.bias'],
              os.path.join(WEIGHTS_DIR, "conv2_b.mem"))
export_bn_params(model_cpu.bn2, "conv2", WEIGHTS_DIR)

# ---- Conv3 ----
print(f"\n  --- Conv3 ({CONV3_IN_CH}→{CONV3_OUT_CH}, 3×3, pad=1, NO pool) ---")
export_conv2d_weights(state_dict['conv3.weight'],
                      os.path.join(WEIGHTS_DIR, "conv3_w.mem"))
export_biases(state_dict['conv3.bias'],
              os.path.join(WEIGHTS_DIR, "conv3_b.mem"))
export_bn_params(model_cpu.bn3, "conv3", WEIGHTS_DIR)

# ---- Conv4 ----
print(f"\n  --- Conv4 ({CONV4_IN_CH}→{CONV4_OUT_CH}, 3×3, pad=1, NO pool) ---")
export_conv2d_weights(state_dict['conv4.weight'],
                      os.path.join(WEIGHTS_DIR, "conv4_w.mem"))
export_biases(state_dict['conv4.bias'],
              os.path.join(WEIGHTS_DIR, "conv4_b.mem"))
export_bn_params(model_cpu.bn4, "conv4", WEIGHTS_DIR)

# ---- FC1 ----
print(f"\n  --- FC1 ({FC1_IN}→{FC1_OUT}) ---")
export_fc_weights(state_dict['fc1.weight'], FC1_IN,
                  os.path.join(WEIGHTS_DIR, "fc1_w.mem"))
export_biases(state_dict['fc1.bias'],
              os.path.join(WEIGHTS_DIR, "fc1_b.mem"))
export_bn_params(model_cpu.bn5, "fc1", WEIGHTS_DIR)

# ---- FC2 ----
print(f"\n  --- FC2 ({FC1_OUT}→{FC2_OUT}, no BN, no dropout) ---")
export_fc_weights(state_dict['fc2.weight'], FC1_OUT,
                  os.path.join(WEIGHTS_DIR, "fc2_w.mem"))
export_biases(state_dict['fc2.bias'],
              os.path.join(WEIGHTS_DIR, "fc2_b.mem"))

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


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*65}")
print(f"  EXPORT COMPLETE")
print(f"{'='*65}")
print(f"  Training method  : Knowledge Distillation (ResNet-18 teacher)")
print(f"  Architecture     : 4 conv layers + GAP + 2 FC")
print(f"  Student accuracy : {best_acc:.2f}%")
print(f"  Total parameters : {total_params:,}")
print(f"  Weight format    : Q16.16 (32-bit hex)")
print(f"  BN parameters    : separate bn_scale + bn_shift per channel")
print(f"  Output directory : {WEIGHTS_DIR}/")
print(f"  Files generated  :")
for fname in sorted(os.listdir(WEIGHTS_DIR)):
    fpath = os.path.join(WEIGHTS_DIR, fname)
    if os.path.isfile(fpath):
        print(f"    {fname:<28} {os.path.getsize(fpath):>8,} bytes")
print(f"{'='*65}")
