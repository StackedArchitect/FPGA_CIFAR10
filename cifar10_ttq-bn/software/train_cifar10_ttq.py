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
# Trained Ternary Quantization (TTQ) + BatchNorm 2D CNN for CIFAR-10
#
# Architecture (identical to baseline v4):
#   Conv1 -> BN1 -> ReLU -> Pool1      [3->32, 3×3, pad=1, pool 2×2]
#   Conv2 -> BN2 -> ReLU -> Pool2      [32->64, 3×3, pad=1, pool 2×2]
#   Conv3 -> BN3 -> ReLU               [64->64, 3×3, pad=1, NO pool]
#   Conv4 -> BN4 -> ReLU               [64->64, 3×3, pad=1, NO pool]
#   Global Average Pool (8×8 -> 1×1)
#   FC1   -> BN5 -> ReLU -> Dropout    [64->256]
#   FC2   -> logits                    [256->10, no BN, no dropout]
#
# TTQ applied to ALL 6 weight layers (conv1–4, fc1–2).
# Each layer has learnable positive scalars Wp, Wn.
# Ternary quantization: w_q = Wp*(W>Δ) - Wn*(W<-Δ), Δ = 0.05*max(|W|)
# Straight-Through Estimator (STE) for backward pass.
#
# Training Strategy (3-Phase):
#   Phase 1:  Teacher (ResNet-18, 200 epochs)
#   Phase 2a: Full-precision warm-up with KD (200 epochs, NO TTQ)
#   Phase 2b: TTQ fine-tuning from warm-up checkpoint (400 epochs)
#
# The warm-up phase lets the model develop large, meaningful latent weights
# BEFORE quantization noise is introduced. This is the critical difference
# vs training TTQ from scratch (which only reached 80%).
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

# ---- TTQ Hyperparameters ----
TTQ_THRESHOLD  = 0.05   # Δ = t * max(|W|) per layer (paper eq. 9)


# ===========================================================================
# TTQ Autograd Function with Straight-Through Estimator (STE)
# ===========================================================================
class TTQFunction(torch.autograd.Function):
    """
    Trained Ternary Quantization forward/backward.

    Forward:
        Delta = TTQ_THRESHOLD * max(|W|)    [paper eq. 9, uses MAX not mean]
        w_q = Wp * (W > Delta) - Wn * (W < -Delta)

    Backward (STE):
        grad_W: passed through from upstream (straight-through)
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
        # STE: pass gradient through to latent weights
        grad_W = grad_output.clone()
        # Gradient w.r.t. Wp: sum of upstream grads where W > Delta
        grad_Wp = (grad_output * pos_mask).sum().unsqueeze(0)
        # Gradient w.r.t. Wn: negative sum of upstream grads where W < -Delta
        grad_Wn = (-grad_output * neg_mask).sum().unsqueeze(0)
        return grad_W, grad_Wp, grad_Wn


def ttq_quantize(W, Wp, Wn):
    """Apply TTQ quantization using the autograd function."""
    return TTQFunction.apply(W, Wp, Wn)


# ===========================================================================
# Student Model — 4-layer CNN with BatchNorm + TTQ
# ===========================================================================
class CIFAR10_CNN2D_TTQ_BN(nn.Module):
    """
    TTQ 2D CNN with BatchNorm for CIFAR-10 (student, v4).
    Architecture (identical structure to baseline CIFAR10_CNN2D_BN):
        Conv1(3->32)  -> BN1 -> ReLU -> MaxPool(2×2)   → 16×16×32
        Conv2(32->64) -> BN2 -> ReLU -> MaxPool(2×2)   → 8×8×64
        Conv3(64->64) -> BN3 -> ReLU                    → 8×8×64
        Conv4(64->64) -> BN4 -> ReLU                    → 8×8×64
        GlobalAvgPool(8×8 -> 1×1)                       → 64
        FC1(64->256)  -> BN5 -> ReLU -> Dropout(0.3)    → 256
        FC2(256->10)  -> logits                          → 10

    TTQ is applied to all 6 weight layers when self.use_ttq == True.
    During warm-up phase, set self.use_ttq = False to use full-precision.
    """
    def __init__(self):
        super().__init__()
        # Convolutional layers (latent full-precision weights)
        self.conv1 = nn.Conv2d(INPUT_CH,    CONV1_OUT_CH, CONV1_KERNEL, padding=CONV1_PAD)
        self.conv2 = nn.Conv2d(CONV2_IN_CH, CONV2_OUT_CH, CONV2_KERNEL, padding=CONV2_PAD)
        self.conv3 = nn.Conv2d(CONV3_IN_CH, CONV3_OUT_CH, CONV3_KERNEL, padding=CONV3_PAD)
        self.conv4 = nn.Conv2d(CONV4_IN_CH, CONV4_OUT_CH, CONV4_KERNEL, padding=CONV4_PAD)

        # Fully connected layers (latent full-precision weights)
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

        # TTQ mode flag — set to False during warm-up phase
        self.use_ttq = True

        # He (Kaiming) initialization for latent weights
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

        # TTQ learnable scalars: Wp and Wn for each weight layer
        # These are re-initialized from warm-up weights before TTQ phase
        self.conv1_wp = nn.Parameter(self.conv1.weight.abs().mean().unsqueeze(0))
        self.conv1_wn = nn.Parameter(self.conv1.weight.abs().mean().unsqueeze(0))
        self.conv2_wp = nn.Parameter(self.conv2.weight.abs().mean().unsqueeze(0))
        self.conv2_wn = nn.Parameter(self.conv2.weight.abs().mean().unsqueeze(0))
        self.conv3_wp = nn.Parameter(self.conv3.weight.abs().mean().unsqueeze(0))
        self.conv3_wn = nn.Parameter(self.conv3.weight.abs().mean().unsqueeze(0))
        self.conv4_wp = nn.Parameter(self.conv4.weight.abs().mean().unsqueeze(0))
        self.conv4_wn = nn.Parameter(self.conv4.weight.abs().mean().unsqueeze(0))
        self.fc1_wp   = nn.Parameter(self.fc1.weight.abs().mean().unsqueeze(0))
        self.fc1_wn   = nn.Parameter(self.fc1.weight.abs().mean().unsqueeze(0))
        self.fc2_wp   = nn.Parameter(self.fc2.weight.abs().mean().unsqueeze(0))
        self.fc2_wn   = nn.Parameter(self.fc2.weight.abs().mean().unsqueeze(0))

    def _get_weight(self, layer, wp, wn):
        """Get weight — full precision or TTQ depending on mode."""
        if self.use_ttq:
            return ttq_quantize(layer.weight, wp, wn)
        else:
            return layer.weight

    def forward(self, x):
        # Conv1 + BN1 + ReLU + Pool
        w = self._get_weight(self.conv1, self.conv1_wp, self.conv1_wn)
        x = F.conv2d(x, w, self.conv1.bias, padding=1)
        x = self.relu(self.bn1(x))
        x = self.pool(x)                           # (B, 32, 16, 16)

        # Conv2 + BN2 + ReLU + Pool
        w = self._get_weight(self.conv2, self.conv2_wp, self.conv2_wn)
        x = F.conv2d(x, w, self.conv2.bias, padding=1)
        x = self.relu(self.bn2(x))
        x = self.pool2(x)                          # (B, 64, 8, 8)

        # Conv3 + BN3 + ReLU (no pool)
        w = self._get_weight(self.conv3, self.conv3_wp, self.conv3_wn)
        x = F.conv2d(x, w, self.conv3.bias, padding=1)
        x = self.relu(self.bn3(x))                 # (B, 64, 8, 8)

        # Conv4 + BN4 + ReLU (no pool)
        w = self._get_weight(self.conv4, self.conv4_wp, self.conv4_wn)
        x = F.conv2d(x, w, self.conv4.bias, padding=1)
        x = self.relu(self.bn4(x))                 # (B, 64, 8, 8)

        # Global Average Pool
        x = self.gap(x)                            # (B, 64, 1, 1)
        x = x.view(-1, FC1_IN)                     # (B, 64)

        # FC1 + BN5 + ReLU + Dropout
        w = self._get_weight(self.fc1, self.fc1_wp, self.fc1_wn)
        x = F.linear(x, w, self.fc1.bias)
        x = self.relu(self.bn5(x))                 # (B, 256)
        x = self.dropout(x)

        # FC2 (no BN, no dropout)
        w = self._get_weight(self.fc2, self.fc2_wp, self.fc2_wn)
        x = F.linear(x, w, self.fc2.bias)
        return x                                    # (B, 10)

    def reinit_ttq_scalars(self):
        """
        Re-initialize Wp/Wn from current (warmed-up) weight statistics.
        Called at the start of TTQ fine-tuning phase.
        Uses mean of positive weights for Wp, mean of negative weights for Wn
        (more informative than raw mean(|W|) after warm-up).
        """
        for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
            layer = getattr(self, name)
            W = layer.weight.detach()
            delta = TTQ_THRESHOLD * W.abs().max()
            pos_weights = W[W > delta]
            neg_weights = W[W < -delta]
            wp_val = pos_weights.abs().mean() if pos_weights.numel() > 0 else W.abs().mean()
            wn_val = neg_weights.abs().mean() if neg_weights.numel() > 0 else W.abs().mean()
            getattr(self, f'{name}_wp').data.fill_(wp_val.item())
            getattr(self, f'{name}_wn').data.fill_(wn_val.item())


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
# Mixup augmentation — helps full-precision warm-up generalise better
# ===========================================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup: convex combination of random pairs of images and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion_fn, pred, y_a, y_b, lam, teacher_logits=None, T=KD_TEMPERATURE, alpha=KD_ALPHA):
    """KD loss with mixup: interpolate CE part, keep KL part intact."""
    if teacher_logits is not None:
        ce_a = F.cross_entropy(pred, y_a)
        ce_b = F.cross_entropy(pred, y_b)
        ce = lam * ce_a + (1 - lam) * ce_b
        soft_student = F.log_softmax(pred / T, dim=1)
        soft_teacher = F.softmax(teacher_logits / T, dim=1).detach()
        kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return alpha * ce + (1.0 - alpha) * T * T * kl
    else:
        return lam * F.cross_entropy(pred, y_a) + (1 - lam) * F.cross_entropy(pred, y_b)


# ===========================================================================
# Setup
# ===========================================================================
print("=" * 70)
print("  CIFAR-10 TTQ+BN CNN — 3-Phase Training")
print("=" * 70)
print(f"  Phase 1  : ResNet-18 teacher (200 epochs)")
print(f"  Phase 2a : Full-precision warm-up + KD (200 epochs, no TTQ)")
print(f"  Phase 2b : TTQ fine-tuning from warm-up + KD (400 epochs)")
print(f"  Teacher  : ResNet-18 (~11M params, ~95%)")
print(f"  Student  : Conv1(3→32)→Conv2(32→64)→Conv3(64→64)→Conv4(64→64)")
print(f"             →GAP→FC1(64→256)→FC2(256→10)")
print(f"  TTQ      : Ternary weights, Δ = {TTQ_THRESHOLD} * max(|W|)")
print(f"  KD       : T={KD_TEMPERATURE}, α={KD_ALPHA}")
print(f"  Target   : ≥90% student accuracy (ternary)")
print("=" * 70)

# ---- Data augmentation (same as baseline) ----
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2470, 0.2435, 0.2616)),
    Cutout(n_holes=1, length=10),
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
TEACHER_PATH  = os.path.join(SCRIPT_DIR, "resnet18_teacher_cifar10.pth")
WARMUP_PATH   = os.path.join(SCRIPT_DIR, "cifar10_warmup_fp_model.pth")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_model.pth")


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
#  PHASE 2a: Full-Precision Warm-Up with KD (NO TTQ)
#
#  Purpose: Let the model learn meaningful latent weights before
#  quantization noise is introduced. This is the critical fix.
#
#  Without warm-up: latent weights stay tiny, quantization Δ is tiny,
#  77% of weights are zero, model underfits at 80%.
#
#  With warm-up: latent weights are large and well-structured,
#  TTQ preserves the model's capacity → 90%+ accuracy.
# ===========================================================================
WARMUP_EPOCHS = 200

student = CIFAR10_CNN2D_TTQ_BN().to(device)
total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

if os.path.exists(WARMUP_PATH):
    print(f"\n[PHASE 2a] Warm-up found — loading {WARMUP_PATH}")
    student.load_state_dict(torch.load(WARMUP_PATH, map_location=device, weights_only=True))
    student.use_ttq = False
    warmup_acc = evaluate(student)
    print(f"           Warm-up accuracy (full-precision): {warmup_acc:.2f}%")
else:
    print(f"\n{'='*70}")
    print(f"  [PHASE 2a] Full-Precision Warm-Up + KD ({WARMUP_EPOCHS} epochs)")
    print(f"             Adam(lr=1e-3, wd=5e-4), CosineAnnealingLR")
    print(f"             Mixup(α=0.2), no TTQ quantization")
    print(f"             Purpose: learn good latent weights before quantization")
    print(f"{'='*70}")
    print(f"{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 44)

    student.use_ttq = False  # CRITICAL: no TTQ during warm-up
    print(f"[INFO] Student parameters: {total_params:,}")
    print(f"[INFO] TTQ mode: OFF (full-precision warm-up)")

    w_optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=5e-4)
    w_scheduler = optim.lr_scheduler.CosineAnnealingLR(w_optimizer, T_max=WARMUP_EPOCHS)

    warmup_best_acc = 0.0
    w_start = time.time()

    warmup_history_loss = []
    warmup_history_train_acc = []
    warmup_history_test_acc  = []

    for epoch in range(WARMUP_EPOCHS):
        student.train()
        student.use_ttq = False
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup augmentation
            mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)

            with torch.no_grad():
                teacher_logits = teacher(mixed_images)

            w_optimizer.zero_grad()
            student_logits = student(mixed_images)
            loss = mixup_criterion(None, student_logits, labels_a, labels_b, lam,
                                   teacher_logits=teacher_logits)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            w_optimizer.step()

            total_loss    += loss.item()
            # For accuracy, use the original (non-mixed) images
            with torch.no_grad():
                preds = student(images).argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        w_scheduler.step()

        train_acc = 100 * total_correct / total_samples
        test_acc  = evaluate(student)
        avg_loss  = total_loss / len(train_loader)

        warmup_history_loss.append(avg_loss)
        warmup_history_train_acc.append(train_acc)
        warmup_history_test_acc.append(test_acc)

        if (epoch + 1) % 10 == 0 or epoch == WARMUP_EPOCHS - 1:
            print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% "
                  f"{test_acc:>9.2f}%", end="")
            if test_acc > warmup_best_acc:
                warmup_best_acc = test_acc
                torch.save(student.state_dict(), WARMUP_PATH)
                print("  <- best saved", end="")
            print()
        elif test_acc > warmup_best_acc:
            warmup_best_acc = test_acc
            torch.save(student.state_dict(), WARMUP_PATH)

    w_time = time.time() - w_start
    print(f"\n[PHASE 2a] Warm-up done in {w_time:.1f}s ({w_time/3600:.2f}h)")
    print(f"           Best full-precision accuracy: {warmup_best_acc:.2f}%")

    # Reload best warm-up model
    student.load_state_dict(torch.load(WARMUP_PATH, map_location=device, weights_only=True))


# ===========================================================================
#  PHASE 2b: TTQ Fine-Tuning from Warm-Up Checkpoint
#
#  Now the latent weights are large and well-structured.
#  TTQ quantization + STE can fine-tune them with minimal accuracy loss.
# ===========================================================================
TTQ_EPOCHS = 400

print(f"\n{'='*70}")
print(f"  [PHASE 2b] TTQ Fine-Tuning + KD ({TTQ_EPOCHS} epochs)")
print(f"             Adam(lr=5e-4, wd=1e-4), CosineAnnealingLR")
print(f"             TTQ: learnable Wp/Wn, Δ = {TTQ_THRESHOLD} * max(|W|)")
print(f"             Initializing Wp/Wn from warm-up weight statistics")
print(f"{'='*70}")

# Re-initialize TTQ scalars from warm-up weights
student.use_ttq = True
student.reinit_ttq_scalars()

print(f"\n[INFO] TTQ scalars initialized from warm-up weights:")
for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
    wp_val = getattr(student, f'{name}_wp').item()
    wn_val = getattr(student, f'{name}_wn').item()
    W = getattr(student, name).weight.detach()
    delta = TTQ_THRESHOLD * W.abs().max().item()
    n_pos = (W > delta).sum().item()
    n_neg = (W < -delta).sum().item()
    n_zero = W.numel() - n_pos - n_neg
    total = W.numel()
    print(f"  {name}: Wp={wp_val:.6f}, Wn={wn_val:.6f}, "
          f"Δ={delta:.6f}, +1={n_pos}({100*n_pos/total:.1f}%) "
          f"0={n_zero}({100*n_zero/total:.1f}%) "
          f"-1={n_neg}({100*n_neg/total:.1f}%)")

# Check pre-quantization accuracy
student.use_ttq = True
ttq_pre_acc = evaluate(student)
print(f"\n[INFO] Accuracy immediately after enabling TTQ: {ttq_pre_acc:.2f}%")
print(f"[INFO] (Expected to drop ~2-5% from full-precision, recovers during fine-tuning)")

print(f"\n{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 44)

# Lower LR for fine-tuning + lower weight decay
s_optimizer = optim.Adam(student.parameters(), lr=5e-4, weight_decay=1e-4)
s_scheduler = optim.lr_scheduler.CosineAnnealingLR(s_optimizer, T_max=TTQ_EPOCHS)

best_acc    = 0.0
s_start     = time.time()

history_loss = []
history_train_acc = []
history_test_acc  = []

for epoch in range(TTQ_EPOCHS):
    student.train()
    student.use_ttq = True
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = teacher(images)

        s_optimizer.zero_grad()
        student_logits = student(images)
        loss = kd_loss(student_logits, teacher_logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
        s_optimizer.step()

        # Clamp Wp/Wn to be positive (they must be positive scalars)
        with torch.no_grad():
            for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
                wp = getattr(student, f'{name}_wp')
                wn = getattr(student, f'{name}_wn')
                wp.data.clamp_(min=1e-7)
                wn.data.clamp_(min=1e-7)

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

    if (epoch + 1) % 10 == 0 or epoch == TTQ_EPOCHS - 1:
        print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% "
              f"{test_acc:>9.2f}%", end="")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), MODEL_SAVE_PATH)
            print("  <- best saved", end="")
        print()
    elif test_acc > best_acc:
        best_acc = test_acc
        torch.save(student.state_dict(), MODEL_SAVE_PATH)

s_time = time.time() - s_start


# ===========================================================================
# Print final TTQ scalar values
# ===========================================================================
student.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
student.eval()
student.use_ttq = True

print(f"\n{'='*70}")
print(f"  TTQ Training Summary")
print(f"{'='*70}")
print(f"  Warm-up time     : Phase 2a elapsed")
print(f"  TTQ FT time      : {s_time:.1f}s ({s_time/3600:.2f}h)")
print(f"  Best test acc    : {best_acc:.2f}%")
print(f"  TTQ epochs       : {TTQ_EPOCHS}")
print(f"  Total parameters : {total_params:,}")
print()

print("  Final TTQ Wp/Wn values:")
for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
    wp_val = getattr(student, f'{name}_wp').item()
    wn_val = getattr(student, f'{name}_wn').item()
    print(f"    {name}: Wp={wp_val:.6f}, Wn={wn_val:.6f}")

# Print ternary weight statistics
print("\n  Ternary weight distribution (best model):")
for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
    layer = getattr(student, name)
    W = layer.weight.detach()
    delta = TTQ_THRESHOLD * W.abs().max()
    n_pos = (W > delta).sum().item()
    n_neg = (W < -delta).sum().item()
    n_zero = W.numel() - n_pos - n_neg
    total = W.numel()
    print(f"    {name:6s}: +1={n_pos:>6} ({100*n_pos/total:5.1f}%)  "
          f" 0={n_zero:>6} ({100*n_zero/total:5.1f}%)  "
          f"-1={n_neg:>6} ({100*n_neg/total:5.1f}%)  "
          f"Δ={delta:.6f}  total={total}")

print(f"\n  Model saved to: {MODEL_SAVE_PATH}")
print(f"{'='*70}")


# ===========================================================================
# Training curve (TTQ phase only)
# ===========================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, TTQ_EPOCHS + 1), history_loss, 'b-', linewidth=1.0)
ax1.set_xlabel('Epoch (TTQ Phase)')
ax1.set_ylabel('Training Loss (KD)')
ax1.set_title('CIFAR-10 TTQ+BN (Phase 2b) — Training Loss')
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, TTQ_EPOCHS + 1), history_train_acc, 'b-', linewidth=1.0,
         label='Train Acc')
ax2.plot(range(1, TTQ_EPOCHS + 1), history_test_acc,  'r-', linewidth=1.0,
         label='Test Acc')
ax2.set_xlabel('Epoch (TTQ Phase)')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('CIFAR-10 TTQ+BN (Phase 2b) — Accuracy')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, "cifar10_ttq_training_curve.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"[INFO] Training curve saved to {plot_path}")
