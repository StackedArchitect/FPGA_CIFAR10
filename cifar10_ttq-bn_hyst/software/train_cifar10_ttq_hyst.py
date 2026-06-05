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

# ===========================================================================
# Trained Ternary Quantization (TTQ) + BatchNorm + Hysteresis Gating for CIFAR-10
#
# # Training Strategy (4-Phase):
#   Phase 1  : ResNet-18 teacher (200 epochs, cached)
#   Phase 2a : Full-precision warm-up + KD + Mixup (200 epochs, no TTQ)
#   Phase 2b : TTQ fine-tuning + KD (400 epochs, standard TTQ)
#   Phase 2c : Hysteresis-aware fine-tuning (100 epochs)
#              Fine-tunes the model with simulated hysteresis masking in the
#              forward path so that it learns to tolerate activation pruning.
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

# ---- KD Hyperparameters ----
KD_TEMPERATURE = 4.0
KD_ALPHA       = 0.3

# ---- TTQ Hyperparameters ----
TTQ_THRESHOLD  = 0.05   # Δ = t * max(|W|) per layer

# Default hysteresis multipliers
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

    neighbor_count = F.conv2d(active_mask, kernel, padding=1, groups=C)
    resolved_uncertain = uncertain_mask * (neighbor_count >= 2.0).float()
    mask = active_mask + resolved_uncertain
    return mask


# ===========================================================================
# Cutout Augmentation
# ===========================================================================
class Cutout:
    def __init__(self, n_holes=1, length=10):
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

    def reinit_ttq_scalars(self):
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


def create_teacher():
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def kd_loss(student_logits, teacher_logits, labels, T=KD_TEMPERATURE, alpha=KD_ALPHA):
    ce = F.cross_entropy(student_logits, labels)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1).detach()
    kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
    return alpha * ce + (1.0 - alpha) * T * T * kl


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, teacher_logits=None, T=KD_TEMPERATURE, alpha=KD_ALPHA):
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
# Calibration helper
# ===========================================================================
def calibrate(model, loader, device):
    model.eval()
    p1_list, p2_list, c3_list = [], [], []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= 5: break
            images = images.to(device)
            w = model._get_weight(model.conv1, model.conv1_wp, model.conv1_wn)
            x = F.conv2d(images, w, model.conv1.bias, padding=1)
            x = model.relu(model.bn1(x))
            x_p1 = model.pool(x)
            
            w = model._get_weight(model.conv2, model.conv2_wp, model.conv2_wn)
            x = F.conv2d(x_p1, w, model.conv2.bias, padding=1)
            x = model.relu(model.bn2(x))
            x_p2 = model.pool2(x)

            w = model._get_weight(model.conv3, model.conv3_wp, model.conv3_wn)
            x = F.conv2d(x_p2, w, model.conv3.bias, padding=1)
            x_c3 = model.relu(model.bn3(x))

            p1_nz = x_p1[x_p1 > 0.0]
            p2_nz = x_p2[x_p2 > 0.0]
            c3_nz = x_c3[x_c3 > 0.0]
            if p1_nz.numel() > 0: p1_list.append(p1_nz.cpu().numpy())
            if p2_nz.numel() > 0: p2_list.append(p2_nz.cpu().numpy())
            if c3_nz.numel() > 0: c3_list.append(c3_nz.cpu().numpy())

    mu1 = np.concatenate(p1_list).mean() if p1_list else 0.8
    mu2 = np.concatenate(p2_list).mean() if p2_list else 1.0
    mu3 = np.concatenate(c3_list).mean() if c3_list else 1.0
    return mu1, mu2, mu3


# ===========================================================================
# Setup and loop
# ===========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
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

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    TEACHER_PATH  = os.path.join(SCRIPT_DIR, "resnet18_teacher_cifar10.pth")
    WARMUP_PATH   = os.path.join(SCRIPT_DIR, "cifar10_warmup_fp_model.pth")
    MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_model.pth")

    student = CIFAR10_CNN2D_TTQ_BN().to(device)

    # 1. Check if model already exists (then we only run Phase 2c)
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"[INFO] Loading converged TTQ model from {MODEL_SAVE_PATH} for Phase 2c Hysteresis Fine-Tuning")
        student.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    else:
        print("[WARNING] Converged model not found. Standard training process would be required.")
        print("          Please compile/run this script once to train standard TTQ before Phase 2c.")

    student.use_ttq = True
    student.eval()

    # Calibrate thresholds
    mu1, mu2, mu3 = calibrate(student, test_loader, device)
    mask1_tl, mask1_th = KL * mu1, KH * mu1
    mask2_tl, mask2_th = KL * mu2, KH * mu2
    mask3_tl, mask3_th = KL * mu3, KH * mu3

    print(f"\n[PHASE 2c] Hysteresis-Aware Fine-Tuning (100 Epochs)...")
    print(f"           Calibrated thresholds: TL1={mask1_tl:.4f}, TH1={mask1_th:.4f}")
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Simple fine-tuning loop for 100 epochs
    best_acc = 0.0
    for epoch in range(100):
        student.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = student.forward_with_pruning(images, use_hysteresis=True,
                                                  mask1_tl=mask1_tl, mask1_th=mask1_th,
                                                  mask2_tl=mask2_tl, mask2_th=mask2_th,
                                                  mask3_tl=mask3_tl, mask3_th=mask3_th)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / total

        # Eval
        student.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = student.forward_with_pruning(images, use_hysteresis=True,
                                                     mask1_tl=mask1_tl, mask1_th=mask1_th,
                                                     mask2_tl=mask2_tl, mask2_th=mask2_th,
                                                     mask3_tl=mask3_tl, mask3_th=mask3_th).argmax(dim=1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_acc = 100.0 * test_correct / test_total
        print(f"  Epoch {epoch+1:>2}/100 | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), MODEL_SAVE_PATH)
            print("  <- Saved best model!")

    print(f"\n[PHASE 2c] Fine-Tuning Complete! Best Hysteresis Accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
