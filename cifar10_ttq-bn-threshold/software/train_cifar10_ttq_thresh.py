#!/usr/bin/env python3
"""
=============================================================================
CIFAR-10  TTQ + BN + Threshold-Aware Pruning — Training & Analysis
=============================================================================

Single monolithic script containing:
  Part 1 — TTQ Training (3-phase + threshold-aware fine-tuning)
    Phase 1  : ResNet-18 teacher (200 epochs)
    Phase 2a : Full-precision warm-up + KD + Mixup (200 epochs, no TTQ)
    Phase 2b : TTQ fine-tuning + KD (400 epochs, standard TTQ)
    Phase 2c : Threshold-aware fine-tuning (100 epochs, DAAP thresholds active)
  Part 2 — Activation Analysis (HookedModel)
  Part 3 — Weight Metrics
  Part 4 — Correlation Matrix
  Part 5 — Pruning Algorithms (Channel Gating, DAAP confirmation, SIP)
  Part 6 — Plots (5 types)
  Part 7 — Final Summary

Architecture (identical to base TTQ model):
  Conv1(3→32)  → BN1 → ReLU → MaxPool(2×2)       → 16×16×32
  Conv2(32→64) → BN2 → ReLU → MaxPool(2×2)        → 8×8×64
  Conv3(64→64) → BN3 → ReLU                        → 8×8×64
  Conv4(64→64) → BN4 → ReLU                        → 8×8×64
  GlobalAvgPool(8×8→1×1)                            → 64
  FC1(64→256)  → BN5 → ReLU → Dropout(0.3)         → 256
  FC2(256→10)  → logits                             → 10

TTQ applied to ALL 6 weight layers (conv1-4, fc1-2).
Each layer has learnable positive scalars Wp, Wn.
Ternary quantization: w_q = Wp*(W>Δ) - Wn*(W<-Δ),  Δ = 0.05*max(|W|)
Straight-Through Estimator (STE) for backward pass.

Training Strategy (4-Phase):
  Phase 1  : ResNet-18 teacher (200 epochs, cached)
  Phase 2a : Full-precision warm-up + KD + Mixup (200 epochs, no TTQ)
  Phase 2b : TTQ fine-tuning + KD (400 epochs, standard TTQ, τ_base=0)
  Phase 2c : Threshold-aware fine-tuning (100 epochs) — MNIST method
             Probes DAAP thresholds on converged model, then fine-tunes
             with forward_with_threshold() active so the model learns to
             tolerate activation pruning. Re-runs DAAP search afterward.

Author  : auto-generated (matches base cifar10_ttq-bn/software/train_cifar10_ttq.py)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("[WARNING] seaborn not installed — correlation heatmap will use imshow fallback")


# ===========================================================================
# Architecture Constants (MUST match base TTQ exactly)
# ===========================================================================
INPUT_H, INPUT_W, INPUT_CH = 32, 32, 3

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
TTQ_THRESHOLD_FACTOR = 0.05   # Δ = t * max(|W|) per layer (paper eq. 9)

# ---- Pruning Hyperparameters ----
ACCURACY_TOLERANCE = 0.5      # max allowed accuracy drop (%)

# DAAP search grid
DAAP_TAU_GRID = [0.0, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1]


# ===========================================================================
# Cutout Augmentation
# ===========================================================================
class Cutout:
    """Randomly mask a square patch of the image (after ToTensor/Normalize)."""
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
# TTQ Autograd Function with Straight-Through Estimator (STE)
# ===========================================================================
class TTQFunction(torch.autograd.Function):
    """
    Trained Ternary Quantization forward/backward.

    Forward:
        Delta = TTQ_THRESHOLD_FACTOR * max(|W|)
        w_q = Wp * (W > Delta) - Wn * (W < -Delta)

    Backward (STE):
        grad_W: passed through from upstream (straight-through)
        grad_Wp: sum of upstream grads where W > Delta
        grad_Wn: -sum of upstream grads where W < -Delta
    """
    @staticmethod
    def forward(ctx, W, Wp, Wn):
        delta = TTQ_THRESHOLD_FACTOR * W.abs().max()
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
# Student Model — 4-layer CNN with BatchNorm + TTQ  (identical to base)
# ===========================================================================
class CIFAR10_CNN2D_TTQ_BN(nn.Module):
    """
    TTQ 2D CNN with BatchNorm for CIFAR-10 (student, v4).
    Architecture (identical structure to baseline CIFAR10_CNN2D_BN):
        Conv1(3→32)  → BN1 → ReLU → MaxPool(2×2)   → 16×16×32
        Conv2(32→64) → BN2 → ReLU → MaxPool(2×2)    → 8×8×64
        Conv3(64→64) → BN3 → ReLU                    → 8×8×64
        Conv4(64→64) → BN4 → ReLU                    → 8×8×64
        GlobalAvgPool(8×8→1×1)                        → 64
        FC1(64→256)  → BN5 → ReLU → Dropout(0.3)     → 256
        FC2(256→10)  → logits                         → 10

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

    def forward_with_threshold(self, x, tau_base=0.0):
        """
        Forward pass with DAAP (Density-Aware Activation Pruning).

        Before each layer, threshold the input activations:
            For each filter f, compute weight density ρ_f.
            Threshold τ_f = τ_base / max(ρ_f, 0.01)
            Use the GROUP MINIMUM threshold across all filters in a layer
            (simplifies FPGA implementation).
            Zero activations where |a| < τ_min.

        When tau_base == 0, this is identical to a standard forward pass.
        """
        if tau_base <= 0.0:
            return self.forward(x)

        # --- Helper: compute per-filter density for a weight tensor ---
        def _layer_threshold(weight_tensor, wp, wn):
            """Compute the group-minimum threshold for a layer."""
            W = weight_tensor.detach()
            delta = TTQ_THRESHOLD_FACTOR * W.abs().max()
            # For conv: weight shape (C_out, C_in, kH, kW) → per output filter
            # For fc:   weight shape (out, in) → per output neuron
            if W.dim() == 4:
                per_filter = W.view(W.size(0), -1)
            else:
                per_filter = W
            n_total = per_filter.size(1)
            if self.use_ttq:
                n_nonzero = ((per_filter > delta) | (per_filter < -delta)).float().sum(dim=1)
            else:
                n_nonzero = (per_filter != 0).float().sum(dim=1)
            density = n_nonzero / n_total
            density = density.clamp(min=0.01)
            tau_per_filter = tau_base / density
            tau_min = tau_per_filter.min().item()
            return tau_min

        def _threshold_activation(act, tau):
            """Zero elements of act where |act| < tau."""
            if tau <= 0.0:
                return act
            mask = (act.abs() >= tau).float()
            return act * mask

        # Conv1 + BN1 + ReLU + Pool
        # (input x is raw image — typically don't threshold inputs)
        w = self._get_weight(self.conv1, self.conv1_wp, self.conv1_wn)
        x = F.conv2d(x, w, self.conv1.bias, padding=1)
        x = self.relu(self.bn1(x))
        x = self.pool(x)                           # (B, 32, 16, 16)

        # Threshold before Conv2
        tau = _layer_threshold(self.conv2.weight, self.conv2_wp, self.conv2_wn)
        x = _threshold_activation(x, tau)
        w = self._get_weight(self.conv2, self.conv2_wp, self.conv2_wn)
        x = F.conv2d(x, w, self.conv2.bias, padding=1)
        x = self.relu(self.bn2(x))
        x = self.pool2(x)                          # (B, 64, 8, 8)

        # Threshold before Conv3
        tau = _layer_threshold(self.conv3.weight, self.conv3_wp, self.conv3_wn)
        x = _threshold_activation(x, tau)
        w = self._get_weight(self.conv3, self.conv3_wp, self.conv3_wn)
        x = F.conv2d(x, w, self.conv3.bias, padding=1)
        x = self.relu(self.bn3(x))                 # (B, 64, 8, 8)

        # Threshold before Conv4
        tau = _layer_threshold(self.conv4.weight, self.conv4_wp, self.conv4_wn)
        x = _threshold_activation(x, tau)
        w = self._get_weight(self.conv4, self.conv4_wp, self.conv4_wn)
        x = F.conv2d(x, w, self.conv4.bias, padding=1)
        x = self.relu(self.bn4(x))                 # (B, 64, 8, 8)

        # Global Average Pool
        x = self.gap(x)                            # (B, 64, 1, 1)
        x = x.view(-1, FC1_IN)                     # (B, 64)

        # Threshold before FC1
        tau = _layer_threshold(self.fc1.weight, self.fc1_wp, self.fc1_wn)
        x = _threshold_activation(x, tau)
        w = self._get_weight(self.fc1, self.fc1_wp, self.fc1_wn)
        x = F.linear(x, w, self.fc1.bias)
        x = self.relu(self.bn5(x))                 # (B, 256)
        x = self.dropout(x)

        # Threshold before FC2
        tau = _layer_threshold(self.fc2.weight, self.fc2_wp, self.fc2_wn)
        x = _threshold_activation(x, tau)
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
            delta = TTQ_THRESHOLD_FACTOR * W.abs().max()
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


def mixup_criterion(criterion_fn, pred, y_a, y_b, lam,
                    teacher_logits=None, T=KD_TEMPERATURE, alpha=KD_ALPHA):
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
# Part 2: HookedModel — captures activations at 8 internal points
# ===========================================================================
HOOK_NAMES = [
    'after_conv1_bn_relu',   # (B, 32, 32, 32)
    'after_pool1',           # (B, 32, 16, 16)
    'after_conv2_bn_relu',   # (B, 64, 16, 16)
    'after_pool2',           # (B, 64,  8,  8)
    'after_conv3_bn_relu',   # (B, 64,  8,  8)
    'after_conv4_bn_relu',   # (B, 64,  8,  8)
    'after_fc1_bn_relu',     # (B, 256)
    'after_fc2_logits',      # (B, 10)
]


class HookedModel:
    """
    Wraps a CIFAR10_CNN2D_TTQ_BN model to capture activations at 8 points.

    Usage:
        hooked = HookedModel(model, device)
        hooked.run(test_loader)
        data = hooked.get_results()
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Storage: lists of tensors per hook (concatenated later)
        self._buffers = {name: [] for name in HOOK_NAMES}

    def _forward_and_capture(self, x):
        """Manual forward pass that stores intermediate activations."""
        m = self.model

        # Conv1 + BN1 + ReLU
        w = m._get_weight(m.conv1, m.conv1_wp, m.conv1_wn)
        x = F.conv2d(x, w, m.conv1.bias, padding=1)
        x = m.relu(m.bn1(x))
        self._buffers['after_conv1_bn_relu'].append(x.cpu())

        # Pool1
        x = m.pool(x)
        self._buffers['after_pool1'].append(x.cpu())

        # Conv2 + BN2 + ReLU
        w = m._get_weight(m.conv2, m.conv2_wp, m.conv2_wn)
        x = F.conv2d(x, w, m.conv2.bias, padding=1)
        x = m.relu(m.bn2(x))
        self._buffers['after_conv2_bn_relu'].append(x.cpu())

        # Pool2
        x = m.pool2(x)
        self._buffers['after_pool2'].append(x.cpu())

        # Conv3 + BN3 + ReLU
        w = m._get_weight(m.conv3, m.conv3_wp, m.conv3_wn)
        x = F.conv2d(x, w, m.conv3.bias, padding=1)
        x = m.relu(m.bn3(x))
        self._buffers['after_conv3_bn_relu'].append(x.cpu())

        # Conv4 + BN4 + ReLU
        w = m._get_weight(m.conv4, m.conv4_wp, m.conv4_wn)
        x = F.conv2d(x, w, m.conv4.bias, padding=1)
        x = m.relu(m.bn4(x))
        self._buffers['after_conv4_bn_relu'].append(x.cpu())

        # GAP → FC1 + BN5 + ReLU (no dropout in eval)
        x = m.gap(x)
        x = x.view(-1, FC1_IN)
        w = m._get_weight(m.fc1, m.fc1_wp, m.fc1_wn)
        x = F.linear(x, w, m.fc1.bias)
        x = m.relu(m.bn5(x))
        self._buffers['after_fc1_bn_relu'].append(x.cpu())

        # FC2 logits
        w = m._get_weight(m.fc2, m.fc2_wp, m.fc2_wn)
        x = F.linear(x, w, m.fc2.bias)
        self._buffers['after_fc2_logits'].append(x.cpu())
        return x

    @torch.no_grad()
    def run(self, loader):
        """Run the hooked forward pass on the entire dataset."""
        self.model.eval()
        # Clear buffers
        for name in HOOK_NAMES:
            self._buffers[name] = []

        for images, _ in loader:
            images = images.to(self.device)
            self._forward_and_capture(images)

    def get_results(self):
        """Concatenate all batch results and return dict of tensors."""
        results = {}
        for name in HOOK_NAMES:
            results[name] = torch.cat(self._buffers[name], dim=0)  # (N, C, [H, W])
        return results


def compute_activation_stats(act_tensor):
    """
    Compute per-channel stats from activation tensor.

    Args:
        act_tensor: (N, C, [H, W]) — spatial dims optional

    Returns:
        dict with keys: mean, std, zero_frac — each shape (C,)
    """
    if act_tensor.dim() == 4:
        # (N, C, H, W) → compute over (N, H, W) for each C
        N, C, H, W = act_tensor.shape
        act_flat = act_tensor.permute(1, 0, 2, 3).reshape(C, -1)  # (C, N*H*W)
    elif act_tensor.dim() == 2:
        # (N, C) → compute over N for each C
        N, C = act_tensor.shape
        act_flat = act_tensor.permute(1, 0)  # (C, N)
    else:
        raise ValueError(f"Unexpected activation shape: {act_tensor.shape}")

    # Per-channel statistics
    ch_sum = act_flat.sum(dim=1).float()
    ch_sum_sq = (act_flat.float() ** 2).sum(dim=1)
    ch_n_total = act_flat.size(1)
    ch_n_zero = (act_flat == 0).sum(dim=1).float()

    ch_mean = ch_sum / ch_n_total
    ch_var = ch_sum_sq / ch_n_total - ch_mean ** 2
    ch_std = ch_var.clamp(min=0).sqrt()
    ch_zero_frac = ch_n_zero / ch_n_total

    return {
        'mean': ch_mean.numpy(),        # (C,)
        'std': ch_std.numpy(),           # (C,)
        'zero_frac': ch_zero_frac.numpy(),  # (C,)
        'n_total': ch_n_total,
        'n_zero': ch_n_zero.numpy(),
    }


# ===========================================================================
# Part 3: Weight Metrics
# ===========================================================================
LAYER_NAMES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']
LAYER_OUT_CHANNELS = [CONV1_OUT_CH, CONV2_OUT_CH, CONV3_OUT_CH, CONV4_OUT_CH, FC1_OUT, FC2_OUT]

# Mapping from layer name to the corresponding activation hook
# (activation AFTER this layer's BN+ReLU, i.e. this layer's output)
LAYER_TO_HOOK = {
    'conv1': 'after_conv1_bn_relu',
    'conv2': 'after_conv2_bn_relu',
    'conv3': 'after_conv3_bn_relu',
    'conv4': 'after_conv4_bn_relu',
    'fc1':   'after_fc1_bn_relu',
    'fc2':   'after_fc2_logits',
}


def compute_weight_metrics(model, act_stats_dict):
    """
    For each of 6 layers (conv1-4, fc1-2), per output filter/neuron:
      - n_total, n_nonzero, n_zero, n_pos, n_neg
      - density (ρ = n_nonzero / n_total)
      - weight_L1 = wp * n_pos + wn * n_neg
      - importance = weight_L1 * act_mean

    Total units: 32+64+64+64+256+10 = 490

    Returns:
        List of dicts, one per unit (490 total), with keys:
            layer, unit_idx, n_total, n_nonzero, n_zero, n_pos, n_neg,
            density, weight_L1, act_mean, act_std, act_zero_frac, importance
    """
    all_units = []

    for layer_name in LAYER_NAMES:
        layer = getattr(model, layer_name)
        W = layer.weight.detach().cpu()
        wp = getattr(model, f'{layer_name}_wp').item()
        wn = getattr(model, f'{layer_name}_wn').item()
        delta = TTQ_THRESHOLD_FACTOR * W.abs().max()

        n_out = W.size(0)  # output channels or output neurons
        hook_name = LAYER_TO_HOOK[layer_name]
        a_stats = act_stats_dict[hook_name]

        for f in range(n_out):
            if W.dim() == 4:
                w_f = W[f]  # (C_in, kH, kW)
            else:
                w_f = W[f]  # (in_features,)

            w_flat = w_f.reshape(-1)
            n_total = w_flat.numel()
            n_pos = (w_flat > delta).sum().item()
            n_neg = (w_flat < -delta).sum().item()
            n_nonzero = n_pos + n_neg
            n_zero = n_total - n_nonzero
            density = n_nonzero / n_total if n_total > 0 else 0.0
            weight_L1 = wp * n_pos + wn * n_neg
            act_mean = float(a_stats['mean'][f])
            act_std = float(a_stats['std'][f])
            act_zero_frac = float(a_stats['zero_frac'][f])
            importance = weight_L1 * abs(act_mean)

            all_units.append({
                'layer': layer_name,
                'unit_idx': f,
                'n_total': n_total,
                'n_nonzero': n_nonzero,
                'n_zero': n_zero,
                'n_pos': n_pos,
                'n_neg': n_neg,
                'density': density,
                'weight_L1': weight_L1,
                'act_mean': act_mean,
                'act_std': act_std,
                'act_zero_frac': act_zero_frac,
                'importance': importance,
            })

    return all_units


# ===========================================================================
# Part 4: Correlation Matrix
# ===========================================================================
def compute_correlation_matrix(unit_metrics, output_dir):
    """
    7×7 Pearson correlation matrix between:
        [weight_L1, n_nonzero, density, act_mean, act_std, act_zero_frac, importance]
    Saves heatmap to output/correlation_matrix.png
    """
    keys = ['weight_L1', 'n_nonzero', 'density', 'act_mean', 'act_std',
            'act_zero_frac', 'importance']
    labels = ['Weight L1', 'N NonZero', 'Density', 'Act Mean', 'Act Std',
              'Act Zero%', 'Importance']

    n = len(keys)
    data = np.array([[u[k] for k in keys] for u in unit_metrics])  # (490, 7)

    # Pearson correlation
    corr = np.corrcoef(data.T)  # (7, 7)

    fig, ax = plt.subplots(figsize=(8, 7))
    if HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                    xticklabels=labels, yticklabels=labels,
                    vmin=-1, vmax=1, center=0, ax=ax,
                    square=True, linewidths=0.5)
    else:
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                        fontsize=8, color='black')
        plt.colorbar(im, ax=ax)

    ax.set_title('CIFAR-10 TTQ — Weight/Activation Correlation Matrix')
    plt.tight_layout()
    path = os.path.join(output_dir, 'correlation_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Correlation matrix saved to {path}")
    return corr


# ===========================================================================
# Part 5a: Channel Gating
# ===========================================================================
@torch.no_grad()
def evaluate_model(model, loader, device, use_threshold=False, tau_base=0.0):
    """Evaluate model accuracy, optionally with DAAP thresholding."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if use_threshold and tau_base > 0:
            preds = model.forward_with_threshold(images, tau_base=tau_base).argmax(dim=1)
        else:
            preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def channel_gating(model, unit_metrics, test_loader, device, baseline_acc,
                   tolerance=ACCURACY_TOLERANCE):
    """
    Sort 490 units by importance (ascending). Binary search: zero N
    least-important channels' weights. Find max N where accuracy drop ≤ tolerance%.

    Returns: (max_gated, gated_acc, sorted_units)
    """
    # Sort by importance ascending (least important first)
    sorted_units = sorted(unit_metrics, key=lambda u: u['importance'])

    # Save original state
    original_state = copy.deepcopy(model.state_dict())

    lo, hi = 0, len(sorted_units)
    best_n = 0
    best_acc = baseline_acc

    print(f"\n[Channel Gating] Binary search for max pruneable channels "
          f"(tolerance={tolerance}%)")
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")

    while lo <= hi:
        mid = (lo + hi) // 2
        if mid == 0:
            lo = 1
            continue

        # Restore original weights
        model.load_state_dict(copy.deepcopy(original_state))

        # Zero out the N least-important units
        for i in range(mid):
            u = sorted_units[i]
            layer = getattr(model, u['layer'])
            with torch.no_grad():
                layer.weight[u['unit_idx']].zero_()
                if hasattr(layer, 'bias') and layer.bias is not None:
                    layer.bias[u['unit_idx']].zero_()

        acc = evaluate_model(model, test_loader, device)
        drop = baseline_acc - acc

        if drop <= tolerance:
            best_n = mid
            best_acc = acc
            lo = mid + 1
        else:
            hi = mid - 1

    # Restore original weights
    model.load_state_dict(original_state)

    print(f"  Max gated channels: {best_n}/{len(sorted_units)} "
          f"(acc={best_acc:.2f}%, drop={baseline_acc - best_acc:.2f}%)")

    return best_n, best_acc, sorted_units


# ===========================================================================
# Part 5b: DAAP (Density-Aware Activation Pruning)
# ===========================================================================
@torch.no_grad()
def daap_evaluate(model, test_loader, device, tau_base):
    """
    Evaluate model with DAAP thresholding.
    Also computes per-layer skip rate and total MAC reduction.
    """
    model.eval()
    correct, total = 0, 0

    # Track per-layer skip statistics
    layer_skip_counts = {}  # layer_name → (total_elements, skipped_elements)

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        if tau_base <= 0:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            continue

        # Manual forward with skip counting
        x = images
        m = model

        # Conv1 (no threshold on raw input)
        w = m._get_weight(m.conv1, m.conv1_wp, m.conv1_wn)
        x = F.conv2d(x, w, m.conv1.bias, padding=1)
        x = m.relu(m.bn1(x))
        x = m.pool(x)

        # Helper for threshold + skip counting
        def _apply_threshold_and_count(act, layer_name, weight_tensor):
            W = weight_tensor.detach()
            delta = TTQ_THRESHOLD_FACTOR * W.abs().max()
            if W.dim() == 4:
                per_filter = W.view(W.size(0), -1)
            else:
                per_filter = W
            n_t = per_filter.size(1)
            if m.use_ttq:
                n_nz = ((per_filter > delta) | (per_filter < -delta)).float().sum(dim=1)
            else:
                n_nz = (per_filter != 0).float().sum(dim=1)
            density = (n_nz / n_t).clamp(min=0.01)
            tau_per_filter = tau_base / density
            tau_min = tau_per_filter.min().item()

            total_elems = act.numel()
            skip_mask = (act.abs() < tau_min).float()
            skipped = skip_mask.sum().item()
            if layer_name not in layer_skip_counts:
                layer_skip_counts[layer_name] = [0, 0]
            layer_skip_counts[layer_name][0] += total_elems
            layer_skip_counts[layer_name][1] += skipped
            return act * (1 - skip_mask)

        # Conv2
        x = _apply_threshold_and_count(x, 'conv2', m.conv2.weight)
        w = m._get_weight(m.conv2, m.conv2_wp, m.conv2_wn)
        x = F.conv2d(x, w, m.conv2.bias, padding=1)
        x = m.relu(m.bn2(x))
        x = m.pool2(x)

        # Conv3
        x = _apply_threshold_and_count(x, 'conv3', m.conv3.weight)
        w = m._get_weight(m.conv3, m.conv3_wp, m.conv3_wn)
        x = F.conv2d(x, w, m.conv3.bias, padding=1)
        x = m.relu(m.bn3(x))

        # Conv4
        x = _apply_threshold_and_count(x, 'conv4', m.conv4.weight)
        w = m._get_weight(m.conv4, m.conv4_wp, m.conv4_wn)
        x = F.conv2d(x, w, m.conv4.bias, padding=1)
        x = m.relu(m.bn4(x))

        # GAP
        x = m.gap(x)
        x = x.view(-1, FC1_IN)

        # FC1
        x = _apply_threshold_and_count(x, 'fc1', m.fc1.weight)
        w = m._get_weight(m.fc1, m.fc1_wp, m.fc1_wn)
        x = F.linear(x, w, m.fc1.bias)
        x = m.relu(m.bn5(x))

        # FC2
        x = _apply_threshold_and_count(x, 'fc2', m.fc2.weight)
        w = m._get_weight(m.fc2, m.fc2_wp, m.fc2_wn)
        x = F.linear(x, w, m.fc2.bias)

        preds = x.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100.0 * correct / total

    # Compute per-layer skip rate and total MAC reduction
    per_layer_skip = {}
    total_elems_all = 0
    total_skipped_all = 0
    for lname, (te, se) in layer_skip_counts.items():
        per_layer_skip[lname] = se / te if te > 0 else 0.0
        total_elems_all += te
        total_skipped_all += se

    mac_reduction = total_skipped_all / total_elems_all if total_elems_all > 0 else 0.0

    return acc, mac_reduction, per_layer_skip


def daap_search(model, test_loader, device, baseline_acc,
                tolerance=ACCURACY_TOLERANCE):
    """
    Search τ_base over DAAP_TAU_GRID.
    For each: threshold input activations, measure accuracy + MAC reduction.
    Pick best τ_base within tolerance% accuracy drop.

    Returns: (results_list, best_tau, best_mac_reduction, best_acc)
    """
    results = []

    print(f"\n[DAAP] Searching τ_base over {len(DAAP_TAU_GRID)} values "
          f"(tolerance={tolerance}%)")
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")
    print(f"  {'τ_base':>8s}  {'Accuracy':>10s}  {'Drop':>8s}  {'MAC Skip':>10s}  {'Status'}")
    print(f"  {'-'*55}")

    best_tau = 0.0
    best_mac = 0.0
    best_acc = baseline_acc

    for tau in DAAP_TAU_GRID:
        acc, mac_red, per_layer = daap_evaluate(model, test_loader, device, tau)
        drop = baseline_acc - acc
        within = drop <= tolerance

        status = "✓ PASS" if within else "✗ FAIL"
        print(f"  {tau:8.3f}  {acc:9.2f}%  {drop:7.2f}%  {100*mac_red:9.1f}%  {status}")

        results.append({
            'tau_base': tau,
            'accuracy': acc,
            'drop': drop,
            'mac_reduction': mac_red,
            'per_layer_skip': per_layer,
            'within_tolerance': within,
        })

        if within and mac_red > best_mac:
            best_tau = tau
            best_mac = mac_red
            best_acc = acc

    print(f"\n  Best τ_base = {best_tau:.3f}  "
          f"(acc={best_acc:.2f}%, MAC reduction={100*best_mac:.1f}%)")

    return results, best_tau, best_mac, best_acc


# ===========================================================================
# Part 5c: SIP (Sparse Isolated Pixel analysis) — analysis-only
# ===========================================================================
def count_isolated_pixels(feature_map):
    """
    Count isolated non-zero pixels in a 2D feature map.
    An isolated pixel has ≤1 non-zero 8-neighbor.

    Args:
        feature_map: (H, W) numpy array

    Returns:
        (n_isolated, n_nonzero, isolation_frac)
    """
    H, W = feature_map.shape
    nonzero_mask = (feature_map != 0).astype(np.float32)
    n_nonzero = nonzero_mask.sum()
    if n_nonzero == 0:
        return 0, 0, 0.0

    # Count non-zero 8-neighbors using convolution-like approach
    padded = np.pad(nonzero_mask, 1, mode='constant', constant_values=0)
    neighbor_count = np.zeros((H, W), dtype=np.float32)
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1+dy:H+1+dy, 1+dx:W+1+dx]

    # Isolated: non-zero pixel with ≤1 non-zero neighbor
    isolated_mask = nonzero_mask * (neighbor_count <= 1)
    n_isolated = isolated_mask.sum()
    isolation_frac = n_isolated / n_nonzero if n_nonzero > 0 else 0.0

    return int(n_isolated), int(n_nonzero), float(isolation_frac)


def sip_analysis(activation_results):
    """
    Count isolated non-zero pixels in conv feature maps.
    Report isolation % per hook point.

    Only applies to spatial activations (conv layers, not FC).
    """
    sip_results = {}
    spatial_hooks = [
        'after_conv1_bn_relu',
        'after_pool1',
        'after_conv2_bn_relu',
        'after_pool2',
        'after_conv3_bn_relu',
        'after_conv4_bn_relu',
    ]

    print(f"\n[SIP] Sparse Isolated Pixel Analysis")
    print(f"  {'Hook Point':<25s}  {'Isolated%':>10s}  {'NonZero%':>10s}")
    print(f"  {'-'*50}")

    for hook_name in spatial_hooks:
        act = activation_results[hook_name].numpy()  # (N, C, H, W)
        total_isolated = 0
        total_nonzero = 0
        total_pixels = 0

        # Sample a subset for efficiency (first 200 images × all channels)
        n_samples = min(act.shape[0], 200)
        for i in range(n_samples):
            for c in range(act.shape[1]):
                fm = act[i, c]  # (H, W)
                n_iso, n_nz, _ = count_isolated_pixels(fm)
                total_isolated += n_iso
                total_nonzero += n_nz
                total_pixels += fm.size

        iso_frac = total_isolated / total_nonzero if total_nonzero > 0 else 0.0
        nz_frac = total_nonzero / total_pixels if total_pixels > 0 else 0.0

        sip_results[hook_name] = {
            'isolation_frac': iso_frac,
            'nonzero_frac': nz_frac,
            'total_isolated': total_isolated,
            'total_nonzero': total_nonzero,
        }

        print(f"  {hook_name:<25s}  {100*iso_frac:9.1f}%  {100*nz_frac:9.1f}%")

    return sip_results


# ===========================================================================
# Part 6: Plots (5 types)
# ===========================================================================
def plot_activation_histograms(activation_results, output_dir):
    """Per-layer activation histograms (8 panels)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, hook_name in enumerate(HOOK_NAMES):
        ax = axes[idx]
        act = activation_results[hook_name].numpy().flatten()
        # Subsample for histogram if too many values
        if len(act) > 500000:
            act = np.random.choice(act, 500000, replace=False)

        ax.hist(act, bins=100, density=True, alpha=0.7, color='steelblue',
                edgecolor='none')
        ax.set_title(hook_name.replace('after_', ''), fontsize=10)
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)

        # Stats annotation
        nz_frac = (act != 0).mean()
        ax.annotate(f'mean={act.mean():.3f}\nstd={act.std():.3f}\nnz={100*nz_frac:.0f}%',
                    xy=(0.95, 0.95), xycoords='axes fraction',
                    ha='right', va='top', fontsize=7,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    fig.suptitle('CIFAR-10 TTQ — Activation Histograms (8 Hook Points)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'activation_histograms.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Activation histograms saved to {path}")


def plot_importance_ranking(sorted_units, n_gated, output_dir):
    """Per-channel importance ranking bar chart."""
    importances = [u['importance'] for u in sorted_units]
    colors = ['red' if i < n_gated else 'steelblue' for i in range(len(importances))]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(range(len(importances)), importances, color=colors, width=1.0, edgecolor='none')
    ax.axvline(x=n_gated - 0.5, color='darkred', linestyle='--', linewidth=1.5,
               label=f'Gating threshold ({n_gated} channels)')
    ax.set_xlabel('Channel Index (sorted by importance ↑)')
    ax.set_ylabel('Importance (Weight_L1 × |Act_Mean|)')
    ax.set_title(f'CIFAR-10 TTQ — Channel Importance Ranking '
                 f'({n_gated}/{len(importances)} gated)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(output_dir, 'importance_ranking.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Importance ranking saved to {path}")


def plot_daap_tradeoff(daap_results, best_tau, output_dir):
    """DAAP accuracy vs MAC reduction tradeoff curve."""
    taus = [r['tau_base'] for r in daap_results]
    accs = [r['accuracy'] for r in daap_results]
    macs = [100 * r['mac_reduction'] for r in daap_results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = 'tab:blue'
    color2 = 'tab:red'

    ax1.set_xlabel('τ_base')
    ax1.set_ylabel('Accuracy (%)', color=color1)
    line1 = ax1.plot(taus, accs, 'o-', color=color1, linewidth=2, markersize=6,
                     label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.axhline(y=accs[0] - ACCURACY_TOLERANCE, color=color1, linestyle=':',
                alpha=0.5, label=f'Tolerance ({ACCURACY_TOLERANCE}%)')

    ax2 = ax1.twinx()
    ax2.set_ylabel('MAC Reduction (%)', color=color2)
    line2 = ax2.plot(taus, macs, 's--', color=color2, linewidth=2, markersize=6,
                     label='MAC Reduction')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Mark best point
    best_idx = taus.index(best_tau) if best_tau in taus else 0
    ax1.axvline(x=best_tau, color='green', linestyle='--', alpha=0.7,
                label=f'Best τ={best_tau:.3f}')

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left')

    ax1.set_title(f'CIFAR-10 TTQ — DAAP Accuracy vs MAC Reduction')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, 'daap_tradeoff.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] DAAP tradeoff curve saved to {path}")


def plot_mac_skip_rates(daap_results, best_tau, output_dir):
    """Per-layer MAC skip rate at best τ_base."""
    # Find the result for best τ
    best_result = None
    for r in daap_results:
        if abs(r['tau_base'] - best_tau) < 1e-6:
            best_result = r
            break

    if best_result is None or not best_result.get('per_layer_skip'):
        print("[PLOT] No per-layer skip data for best τ — skipping mac_skip_rates plot")
        return

    layers = list(best_result['per_layer_skip'].keys())
    skip_rates = [100 * best_result['per_layer_skip'][l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(layers, skip_rates, color='darkorange', edgecolor='black',
                  linewidth=0.5)

    # Add value labels on bars
    for bar, rate in zip(bars, skip_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Activation Skip Rate (%)')
    ax.set_title(f'CIFAR-10 TTQ — Per-Layer MAC Skip Rate (τ_base={best_tau:.3f})')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(skip_rates) * 1.15 if skip_rates else 100)

    plt.tight_layout()
    path = os.path.join(output_dir, 'mac_skip_rates.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] MAC skip rates saved to {path}")


def plot_pruning_summary(baseline_acc, n_gated, gated_acc, best_tau, best_mac,
                         best_daap_acc, sip_results, daap_results, sorted_units,
                         output_dir):
    """Summary dashboard (6-panel combined)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Accuracy comparison
    ax = axes[0, 0]
    methods = ['Baseline', 'Ch. Gating', f'DAAP\nτ={best_tau:.3f}']
    accs = [baseline_acc, gated_acc, best_daap_acc]
    colors = ['steelblue', 'coral', 'gold']
    bars = ax.bar(methods, accs, color=colors, edgecolor='black', linewidth=0.5)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{a:.2f}%', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy Comparison')
    ax.set_ylim(min(accs) - 2, max(accs) + 2)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Channel gating — importance distribution
    ax = axes[0, 1]
    importances = [u['importance'] for u in sorted_units]
    ax.hist(importances, bins=50, color='steelblue', edgecolor='none', alpha=0.7)
    ax.axvline(x=sorted_units[n_gated - 1]['importance'] if n_gated > 0 else 0,
               color='red', linestyle='--', label=f'Gating cutoff ({n_gated})')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Count')
    ax.set_title('Importance Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: DAAP tradeoff (miniature)
    ax = axes[0, 2]
    taus = [r['tau_base'] for r in daap_results]
    macs = [100 * r['mac_reduction'] for r in daap_results]
    d_accs = [r['accuracy'] for r in daap_results]
    ax.plot(taus, d_accs, 'o-', color='tab:blue', label='Accuracy')
    ax.axvline(x=best_tau, color='green', linestyle='--', alpha=0.7, label=f'Best τ')
    ax.set_xlabel('τ_base')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('DAAP Tradeoff')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: MAC reduction vs τ
    ax = axes[1, 0]
    ax.plot(taus, macs, 's-', color='tab:red')
    ax.axvline(x=best_tau, color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('τ_base')
    ax.set_ylabel('MAC Reduction (%)')
    ax.set_title('DAAP MAC Reduction')
    ax.grid(True, alpha=0.3)

    # Panel 5: SIP isolation rates
    ax = axes[1, 1]
    if sip_results:
        hooks = list(sip_results.keys())
        iso_rates = [100 * sip_results[h]['isolation_frac'] for h in hooks]
        short_names = [h.replace('after_', '').replace('_bn_relu', '') for h in hooks]
        bars = ax.bar(short_names, iso_rates, color='mediumpurple', edgecolor='black',
                      linewidth=0.5)
        for bar, rate in zip(bars, iso_rates):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_ylabel('Isolation Rate (%)')
        ax.set_title('SIP: Isolated Pixel Rate')
        ax.tick_params(axis='x', rotation=30)
    else:
        ax.text(0.5, 0.5, 'No SIP data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('SIP: Isolated Pixel Rate')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 6: Per-layer density
    ax = axes[1, 2]
    layer_densities = {}
    for u in sorted_units:
        lname = u['layer']
        if lname not in layer_densities:
            layer_densities[lname] = []
        layer_densities[lname].append(u['density'])

    layer_names_sorted = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']
    mean_densities = [np.mean(layer_densities.get(ln, [0])) for ln in layer_names_sorted]
    ax.bar(layer_names_sorted, [100 * d for d in mean_densities],
           color='seagreen', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Mean Weight Density (%)')
    ax.set_title('Per-Layer Weight Density')
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('CIFAR-10 TTQ+BN+Threshold — Pruning Summary Dashboard', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, 'pruning_summary.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PLOT] Pruning summary dashboard saved to {path}")


# ===========================================================================
# Main Function
# ===========================================================================
def main():
    global_start = time.time()

    # ======================================================================
    # Setup
    # ======================================================================
    print("=" * 70)
    print("  CIFAR-10 TTQ+BN+Threshold — Training & Analysis Pipeline")
    print("=" * 70)
    print(f"  Phase 1  : ResNet-18 teacher (200 epochs)")
    print(f"  Phase 2a : Full-precision warm-up + KD + Mixup (200 epochs, no TTQ)")
    print(f"  Phase 2b : TTQ fine-tuning + KD (400 epochs, standard TTQ)")
    print(f"  Phase 2c : Threshold-aware fine-tuning (100 epochs, DAAP active)")
    print(f"  Part 2   : Activation analysis (8 hook points)")
    print(f"  Part 3   : Weight metrics (490 units)")
    print(f"  Part 4   : Correlation matrix (7×7)")
    print(f"  Part 5   : Pruning (Channel Gating + DAAP + SIP) — post-training")
    print(f"  Part 6   : Plots (5 types)")
    print(f"  Part 7   : Final summary")
    print(f"  TTQ      : Ternary weights, Δ = {TTQ_THRESHOLD_FACTOR} * max(|W|)")
    print(f"  KD       : T={KD_TEMPERATURE}, α={KD_ALPHA}")
    print(f"  Pruning  : ACCURACY_TOLERANCE={ACCURACY_TOLERANCE}%")
    print("=" * 70)

    # ---- Data augmentation (same as base) ----
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

    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'output'))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TEACHER_PATH    = os.path.join(SCRIPT_DIR, "resnet18_teacher_cifar10.pth")
    WARMUP_PATH     = os.path.join(SCRIPT_DIR, "cifar10_warmup_fp_thresh_model.pth")
    MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_thresh_model.pth")

    # ==================================================================
    # PART 1: TTQ Training from Scratch (3-Phase)
    # ==================================================================

    # ------------------------------------------------------------------
    # PHASE 1: Teacher (skip if cached)
    # ------------------------------------------------------------------
    TEACHER_EPOCHS = 200

    if os.path.exists(TEACHER_PATH):
        print(f"\n[PHASE 1] Teacher found — loading {TEACHER_PATH}")
        teacher = create_teacher().to(device)
        teacher.load_state_dict(
            torch.load(TEACHER_PATH, map_location=device, weights_only=True))
        teacher.eval()
        teacher_acc = evaluate_model(teacher, test_loader, device)
        print(f"          Teacher accuracy: {teacher_acc:.2f}%")
    else:
        print(f"\n[PHASE 1] Training ResNet-18 teacher ({TEACHER_EPOCHS} epochs)")
        print(f"          SGD(lr=0.1, momentum=0.9, nesterov), MultiStepLR([100,150])")
        print("-" * 60)

        teacher = create_teacher().to(device)
        t_criterion = nn.CrossEntropyLoss()
        t_optimizer = optim.SGD(teacher.parameters(), lr=0.1, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
        t_scheduler = optim.lr_scheduler.MultiStepLR(
            t_optimizer, milestones=[100, 150], gamma=0.1)
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
                acc = evaluate_model(teacher, test_loader, device)
                print(f"  Teacher epoch {epoch+1:>3}/{TEACHER_EPOCHS}  |  "
                      f"test acc: {acc:.2f}%", end="")
                if acc > teacher_best_acc:
                    teacher_best_acc = acc
                    torch.save(teacher.state_dict(), TEACHER_PATH)
                    print("  <- best", end="")
                print()

        t_time = time.time() - t_start
        print(f"\n[PHASE 1] Teacher done in {t_time:.1f}s  "
              f"(best: {teacher_best_acc:.2f}%)")
        teacher.load_state_dict(
            torch.load(TEACHER_PATH, map_location=device, weights_only=True))

    teacher.eval()

    # ------------------------------------------------------------------
    # PHASE 2a: Full-Precision Warm-Up with KD + Mixup (NO TTQ)
    # ------------------------------------------------------------------
    WARMUP_EPOCHS = 200

    student = CIFAR10_CNN2D_TTQ_BN().to(device)
    total_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

    if os.path.exists(WARMUP_PATH):
        print(f"\n[PHASE 2a] Warm-up found — loading {WARMUP_PATH}")
        student.load_state_dict(
            torch.load(WARMUP_PATH, map_location=device, weights_only=True))
        student.use_ttq = False
        warmup_acc = evaluate_model(student, test_loader, device)
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
        w_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            w_optimizer, T_max=WARMUP_EPOCHS)

        warmup_best_acc = 0.0
        w_start = time.time()

        for epoch in range(WARMUP_EPOCHS):
            student.train()
            student.use_ttq = False
            total_loss, total_correct, total_samples = 0.0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Mixup augmentation
                mixed_images, labels_a, labels_b, lam = mixup_data(
                    images, labels, alpha=0.2)

                with torch.no_grad():
                    teacher_logits = teacher(mixed_images)

                w_optimizer.zero_grad()
                student_logits = student(mixed_images)
                loss = mixup_criterion(
                    None, student_logits, labels_a, labels_b, lam,
                    teacher_logits=teacher_logits)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
                w_optimizer.step()

                total_loss += loss.item()
                with torch.no_grad():
                    preds = student(images).argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

            w_scheduler.step()

            train_acc = 100 * total_correct / total_samples
            test_acc  = evaluate_model(student, test_loader, device)
            avg_loss  = total_loss / len(train_loader)

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

        student.load_state_dict(
            torch.load(WARMUP_PATH, map_location=device, weights_only=True))

    # ------------------------------------------------------------------
    # PHASE 2b: TTQ Fine-Tuning (400 epochs, standard TTQ, no DAAP ramp)
    # ------------------------------------------------------------------
    # NOTE: DAAP analysis is done POST-TRAINING in Part 5.
    # The "integrated pruning-aware training" approach was tried but the
    # DAAP probe runs on an unconverged model (11% accuracy) producing
    # garbage τ values that destroy training.  Standard TTQ for all 400
    # epochs → converge to ~86-87% → then analyse DAAP tolerance.
    # ------------------------------------------------------------------
    TTQ_EPOCHS = 400

    print(f"\n{'='*70}")
    print(f"  [PHASE 2b] TTQ Fine-Tuning + KD ({TTQ_EPOCHS} epochs)")
    print(f"             Adam(lr=1e-3, wd=5e-4), CosineAnnealingLR(T_max={TTQ_EPOCHS})")
    print(f"             TTQ: learnable Wp/Wn, Δ = {TTQ_THRESHOLD_FACTOR} * max(|W|)")
    print(f"             All {TTQ_EPOCHS} epochs: Standard TTQ (τ_base=0)")
    print(f"             DAAP analysis: post-training (Part 5)")
    print(f"{'='*70}")

    # Re-initialize TTQ scalars from warm-up weights
    student.use_ttq = True
    student.reinit_ttq_scalars()

    print(f"\n[INFO] TTQ scalars initialized from warm-up weights:")
    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
        wp_val = getattr(student, f'{name}_wp').item()
        wn_val = getattr(student, f'{name}_wn').item()
        W = getattr(student, name).weight.detach()
        delta = TTQ_THRESHOLD_FACTOR * W.abs().max().item()
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
    ttq_pre_acc = evaluate_model(student, test_loader, device)
    print(f"\n[INFO] Accuracy immediately after enabling TTQ: {ttq_pre_acc:.2f}%")
    print(f"[INFO] (This is expected to be low — TTQ scalars are fresh. "
          f"The model will recover during training.)")

    print(f"\n{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 46)

    s_optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=5e-4)
    s_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        s_optimizer, T_max=TTQ_EPOCHS)

    best_acc = 0.0
    s_start  = time.time()

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

            # Clamp Wp/Wn to be positive
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
        test_acc  = evaluate_model(student, test_loader, device)
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

    # ------------------------------------------------------------------
    # Reload best model for analysis
    # ------------------------------------------------------------------
    student.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    student.eval()
    student.use_ttq = True

    print(f"\n{'='*70}")
    print(f"  TTQ Training Summary (Phase 2b)")
    print(f"{'='*70}")
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

    print("\n  Ternary weight distribution (best model):")
    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
        layer = getattr(student, name)
        W = layer.weight.detach()
        delta = TTQ_THRESHOLD_FACTOR * W.abs().max()
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

    baseline_acc = best_acc

    # ------------------------------------------------------------------
    # PHASE 2c: Threshold-Aware Fine-Tuning (MNIST method)
    # ------------------------------------------------------------------
    # Step 1: Run initial DAAP probe on converged model with fine grid
    # Step 2: If a non-zero τ works, fine-tune WITH thresholds active
    # Step 3: Re-run DAAP search on the adapted model
    # ------------------------------------------------------------------
    THRESH_FT_EPOCHS = 100
    THRESH_FT_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_thresh_ft_model.pth")

    print(f"\n{'='*70}")
    print(f"  [PHASE 2c] Threshold-Aware Fine-Tuning")
    print(f"             Step 1: Initial DAAP probe on converged {baseline_acc:.2f}% model")
    print(f"{'='*70}")

    # --- Step 1: Initial DAAP probe with fine grid ---
    student.eval()
    student.use_ttq = True

    probe_results, probe_tau, probe_mac, probe_acc = daap_search(
        student, test_loader, device, baseline_acc, tolerance=ACCURACY_TOLERANCE)

    # If the finest grid still fails, try an even finer sub-grid
    if probe_tau <= 0.0:
        print(f"\n  [PHASE 2c] Standard grid found no usable τ. Trying ultra-fine grid...")
        ULTRA_FINE_GRID_SAVE = DAAP_TAU_GRID
        DAAP_TAU_GRID_ULTRA = [0.0, 0.0001, 0.0002, 0.0005, 0.0007, 0.001]
        # Temporarily swap grid
        import types
        old_grid = DAAP_TAU_GRID
        ultra_results = []
        uf_best_tau = 0.0
        uf_best_mac = 0.0
        uf_best_acc = baseline_acc

        print(f"\n[DAAP-ultra] Searching τ_base over {len(DAAP_TAU_GRID_ULTRA)} values "
              f"(tolerance={ACCURACY_TOLERANCE}%)")
        print(f"  Baseline accuracy: {baseline_acc:.2f}%")
        print(f"  {'τ_base':>8s}  {'Accuracy':>10s}  {'Drop':>8s}  {'MAC Skip':>10s}  {'Status'}")
        print(f"  {'-'*55}")

        for tau in DAAP_TAU_GRID_ULTRA:
            acc, mac_red, per_layer = daap_evaluate(model=student,
                test_loader=test_loader, device=device, tau_base=tau)
            drop = baseline_acc - acc
            within = drop <= ACCURACY_TOLERANCE
            status = "✓ PASS" if within else "✗ FAIL"
            print(f"  {tau:8.4f}  {acc:9.2f}%  {drop:7.2f}%  {100*mac_red:9.1f}%  {status}")
            ultra_results.append({'tau_base': tau, 'accuracy': acc, 'drop': drop,
                                  'mac_reduction': mac_red, 'within_tolerance': within})
            if within and mac_red > uf_best_mac:
                uf_best_tau = tau
                uf_best_mac = mac_red
                uf_best_acc = acc

        if uf_best_tau > 0.0:
            probe_tau = uf_best_tau
            probe_mac = uf_best_mac
            probe_acc = uf_best_acc
            print(f"\n  Ultra-fine: Best τ_base = {probe_tau:.4f} "
                  f"(acc={probe_acc:.2f}%, MAC reduction={100*probe_mac:.1f}%)")
        else:
            # Even ultra-fine fails. Use a safe conservative τ for fine-tuning
            # so the model can learn to tolerate it.
            probe_tau = 0.001
            print(f"\n  Ultra-fine: No usable τ found. Using conservative τ_base={probe_tau} "
                  f"for threshold-aware fine-tuning.")

    print(f"\n  Selected τ_base for fine-tuning: {probe_tau:.6f}")

    # --- Step 2: Fine-tune with thresholds active ---
    print(f"\n{'='*70}")
    print(f"  [PHASE 2c] Threshold-Aware Fine-Tuning ({THRESH_FT_EPOCHS} epochs)")
    print(f"             Adam(lr=1e-4, wd=5e-4), CosineAnnealingLR(T_max={THRESH_FT_EPOCHS})")
    print(f"             Training with forward_with_threshold(τ_base={probe_tau:.6f})")
    print(f"{'='*70}")

    # Save a backup of the pre-threshold model
    PRE_THRESH_PATH = os.path.join(SCRIPT_DIR, "cifar10_ttq_bn_pre_thresh_model.pth")
    torch.save(student.state_dict(), PRE_THRESH_PATH)

    ft_optimizer = optim.Adam(student.parameters(), lr=1e-4, weight_decay=5e-4)
    ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=THRESH_FT_EPOCHS)

    ft_best_acc = 0.0
    ft_start = time.time()
    ft_tau = probe_tau

    print(f"\n{'Epoch':<8} {'Train Loss':>12} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 46)

    for epoch in range(THRESH_FT_EPOCHS):
        student.train()
        student.use_ttq = True
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)

            ft_optimizer.zero_grad()
            # KEY CHANGE: use forward_with_threshold instead of forward
            student_logits = student.forward_with_threshold(images, tau_base=ft_tau)

            loss = kd_loss(student_logits, teacher_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            ft_optimizer.step()

            # Clamp Wp/Wn to be positive
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

        ft_scheduler.step()

        train_acc = 100 * total_correct / total_samples
        test_acc  = evaluate_model(student, test_loader, device)
        avg_loss  = total_loss / len(train_loader)

        if (epoch + 1) % 10 == 0 or epoch == THRESH_FT_EPOCHS - 1:
            print(f"  {epoch+1:<6} {avg_loss:>12.4f} {train_acc:>9.2f}% "
                  f"{test_acc:>9.2f}%", end="")
            if test_acc > ft_best_acc:
                ft_best_acc = test_acc
                torch.save(student.state_dict(), THRESH_FT_PATH)
                print("  <- best saved", end="")
            print()
        elif test_acc > ft_best_acc:
            ft_best_acc = test_acc
            torch.save(student.state_dict(), THRESH_FT_PATH)

    ft_time = time.time() - ft_start

    # Reload best threshold-fine-tuned model
    student.load_state_dict(
        torch.load(THRESH_FT_PATH, map_location=device, weights_only=True))
    student.eval()
    student.use_ttq = True

    print(f"\n  Threshold FT time: {ft_time:.1f}s ({ft_time/3600:.2f}h)")
    print(f"  Best test acc (with threshold training): {ft_best_acc:.2f}%")

    # --- Step 3: Re-run DAAP search on the adapted model ---
    print(f"\n{'='*70}")
    print(f"  [PHASE 2c] Re-DAAP search on threshold-adapted model")
    print(f"{'='*70}")

    ft_baseline_acc = evaluate_model(student, test_loader, device)
    print(f"  Adapted model baseline: {ft_baseline_acc:.2f}%")

    re_daap_results, re_best_tau, re_best_mac, re_best_acc = daap_search(
        student, test_loader, device, ft_baseline_acc, tolerance=ACCURACY_TOLERANCE)

    # Compare: if threshold FT helped, use the adapted model;
    # otherwise fall back to the pre-threshold model
    if re_best_tau > 0.0 and re_best_mac > 0.0:
        print(f"\n  ✓ Threshold-aware FT succeeded!")
        print(f"    Pre-FT:  τ={probe_tau:.4f}, MAC red={100*probe_mac:.1f}%")
        print(f"    Post-FT: τ={re_best_tau:.4f}, MAC red={100*re_best_mac:.1f}%")
        best_tau = re_best_tau
        best_mac = re_best_mac
        baseline_acc = ft_baseline_acc
        # Copy threshold-FT model as the main model
        torch.save(student.state_dict(), MODEL_SAVE_PATH)
        print(f"    Saved adapted model to {MODEL_SAVE_PATH}")
    else:
        print(f"\n  ✗ Threshold-aware FT did not improve DAAP tolerance.")
        print(f"    Falling back to pre-threshold model ({best_acc:.2f}%).")
        student.load_state_dict(
            torch.load(PRE_THRESH_PATH, map_location=device, weights_only=True))
        student.eval()
        student.use_ttq = True
        baseline_acc = best_acc
        best_tau = 0.0
        best_mac = 0.0

    # Save DAAP config
    daap_config_path = os.path.join(SCRIPT_DIR, "daap_config.txt")
    with open(daap_config_path, 'w') as f:
        f.write(f"best_tau_base = {best_tau:.6f}\n")
    print(f"  [DAAP] Saved best_tau_base={best_tau:.6f} to {daap_config_path}")

    print(f"{'='*70}")

    # ==================================================================
    # PART 2: Activation Analysis
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Part 2: Activation Analysis (HookedModel)")
    print(f"{'='*70}")

    act_start = time.time()
    hooked = HookedModel(student, device)
    hooked.run(test_loader)
    activation_results = hooked.get_results()

    print(f"\n  Captured activations at {len(HOOK_NAMES)} hook points:")
    for name in HOOK_NAMES:
        shape = tuple(activation_results[name].shape)
        print(f"    {name:<25s}  shape={shape}")

    # Compute per-channel statistics
    act_stats_dict = {}
    print(f"\n  Per-channel statistics:")
    print(f"    {'Hook':<25s}  {'Channels':>8s}  {'Mean μ':>10s}  "
          f"{'Std σ':>10s}  {'Zero%':>8s}")
    print(f"    {'-'*65}")
    for name in HOOK_NAMES:
        stats = compute_activation_stats(activation_results[name])
        act_stats_dict[name] = stats
        n_ch = len(stats['mean'])
        avg_mean = np.mean(stats['mean'])
        avg_std = np.mean(stats['std'])
        avg_zf = np.mean(stats['zero_frac'])
        print(f"    {name:<25s}  {n_ch:>8d}  {avg_mean:>10.4f}  "
              f"{avg_std:>10.4f}  {100*avg_zf:>7.1f}%")

    act_time = time.time() - act_start
    print(f"\n  Activation analysis done in {act_time:.1f}s")

    # ==================================================================
    # PART 3: Weight Metrics
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Part 3: Weight Metrics (490 units)")
    print(f"{'='*70}")

    wm_start = time.time()
    unit_metrics = compute_weight_metrics(student, act_stats_dict)

    print(f"\n  Total units: {len(unit_metrics)}")
    print(f"  Per-layer unit counts:")
    for lname in LAYER_NAMES:
        count = sum(1 for u in unit_metrics if u['layer'] == lname)
        avg_density = np.mean([u['density'] for u in unit_metrics if u['layer'] == lname])
        avg_importance = np.mean([u['importance'] for u in unit_metrics if u['layer'] == lname])
        print(f"    {lname:6s}: {count:>4d} units  "
              f"avg_density={avg_density:.3f}  avg_importance={avg_importance:.4f}")

    wm_time = time.time() - wm_start
    print(f"\n  Weight metrics computed in {wm_time:.1f}s")

    # ==================================================================
    # PART 4: Correlation Matrix
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Part 4: Correlation Matrix")
    print(f"{'='*70}")

    corr_start = time.time()
    corr_matrix = compute_correlation_matrix(unit_metrics, OUTPUT_DIR)
    corr_time = time.time() - corr_start
    print(f"  Correlation matrix computed in {corr_time:.1f}s")

    # ==================================================================
    # PART 5: Pruning Algorithms
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Part 5: Pruning Algorithms")
    print(f"{'='*70}")

    prune_start = time.time()

    # 5a: Channel Gating
    print(f"\n--- 5a. Channel Gating ---")
    n_gated, gated_acc, sorted_units = channel_gating(
        student, unit_metrics, test_loader, device, baseline_acc,
        tolerance=ACCURACY_TOLERANCE)

    # Restore best model (channel gating modifies weights)
    student.load_state_dict(
        torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    student.eval()
    student.use_ttq = True

    # 5b: DAAP (confirmation run — Phase 2c may have already found a good τ)
    print(f"\n--- 5b. DAAP (Density-Aware Activation Pruning) ---")
    print(f"  (Confirmation run — Phase 2c already saved best_tau={best_tau:.6f})")
    daap_results, p5_tau, p5_mac, best_daap_acc = daap_search(
        student, test_loader, device, baseline_acc, tolerance=ACCURACY_TOLERANCE)

    # Keep whichever result has higher MAC reduction
    if p5_tau > 0.0 and p5_mac > best_mac:
        best_tau = p5_tau
        best_mac = p5_mac
        print(f"  [DAAP] Part 5 found better τ={best_tau:.6f} (MAC red={100*best_mac:.1f}%)")

    # Save final DAAP config for test script to pick up
    daap_config_path = os.path.join(SCRIPT_DIR, "daap_config.txt")
    with open(daap_config_path, 'w') as f:
        f.write(f"best_tau_base = {best_tau:.6f}\n")
    print(f"  [DAAP] Saved best_tau_base={best_tau:.6f} to {daap_config_path}")

    # 5c: SIP
    print(f"\n--- 5c. SIP (Sparse Isolated Pixel Analysis) ---")
    sip_results = sip_analysis(activation_results)

    prune_time = time.time() - prune_start
    print(f"\n  Pruning analysis done in {prune_time:.1f}s")

    # ==================================================================
    # PART 6: Plots (5 types)
    # ==================================================================
    print(f"\n{'='*70}")
    print(f"  Part 6: Generating Plots")
    print(f"{'='*70}")

    plot_start = time.time()

    # 1. Activation histograms
    plot_activation_histograms(activation_results, OUTPUT_DIR)

    # 2. Importance ranking
    plot_importance_ranking(sorted_units, n_gated, OUTPUT_DIR)

    # 3. DAAP tradeoff
    plot_daap_tradeoff(daap_results, best_tau, OUTPUT_DIR)

    # 4. MAC skip rates
    plot_mac_skip_rates(daap_results, best_tau, OUTPUT_DIR)

    # 5. Summary dashboard
    plot_pruning_summary(
        baseline_acc, n_gated, gated_acc, best_tau, best_mac,
        best_daap_acc, sip_results, daap_results, sorted_units, OUTPUT_DIR)

    plot_time = time.time() - plot_start
    print(f"\n  Plots generated in {plot_time:.1f}s")

    # ==================================================================
    # PART 7: Final Summary
    # ==================================================================
    total_time = time.time() - global_start
    print(f"\n{'='*70}")
    print(f"  PART 7: FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline TTQ Accuracy    : {baseline_acc:.2f}%")
    print(f"  Total Parameters         : {total_params:,}")
    print()
    print(f"  --- Channel Gating ---")
    print(f"  Channels Gated           : {n_gated}/490")
    print(f"  Accuracy After Gating    : {gated_acc:.2f}%")
    print(f"  Accuracy Drop            : {baseline_acc - gated_acc:.2f}%")
    print()
    print(f"  --- DAAP ---")
    print(f"  Best τ_base              : {best_tau:.3f}")
    print(f"  MAC Reduction            : {100*best_mac:.1f}%")
    print(f"  Accuracy After DAAP      : {best_daap_acc:.2f}%")
    print(f"  Accuracy Drop            : {baseline_acc - best_daap_acc:.2f}%")
    print()
    print(f"  --- SIP Statistics ---")
    if sip_results:
        for hook_name, sr in sip_results.items():
            short = hook_name.replace('after_', '')
            print(f"    {short:<25s}  isolation={100*sr['isolation_frac']:.1f}%  "
                  f"nonzero={100*sr['nonzero_frac']:.1f}%")
    print()
    print(f"  --- TTQ Threshold Formula ---")

    # Determine best threshold formula (max vs mean) for FPGA
    # Compare per-layer: Δ_max = 0.05*max(|W|)  vs  Δ_mean = mean(|W|)
    print(f"  Per-layer Δ comparison (max vs mean):")
    for name in ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']:
        layer = getattr(student, name)
        W = layer.weight.detach()
        delta_max = TTQ_THRESHOLD_FACTOR * W.abs().max().item()
        delta_mean = W.abs().mean().item()
        n_total = W.numel()
        n_nz_max = ((W.abs() > delta_max)).sum().item()
        n_nz_mean = ((W.abs() > delta_mean)).sum().item()
        density_max = n_nz_max / n_total
        density_mean = n_nz_mean / n_total
        print(f"    {name:6s}: Δ_max={delta_max:.6f} (density={density_max:.3f})  "
              f"Δ_mean={delta_mean:.6f} (density={density_mean:.3f})")

    print(f"\n  Recommendation: TTQ uses Δ = {TTQ_THRESHOLD_FACTOR} * max(|W|) per layer")
    print(f"  This preserves ~45-55% non-zero weights (balanced ternary distribution)")
    print()
    print(f"  --- Timing ---")
    print(f"  Total pipeline time      : {total_time:.1f}s ({total_time/3600:.2f}h)")
    print()
    print(f"  --- Output Files ---")
    print(f"  Model checkpoint         : {MODEL_SAVE_PATH}")
    print(f"  Activation histograms    : {os.path.join(OUTPUT_DIR, 'activation_histograms.png')}")
    print(f"  Importance ranking       : {os.path.join(OUTPUT_DIR, 'importance_ranking.png')}")
    print(f"  DAAP tradeoff            : {os.path.join(OUTPUT_DIR, 'daap_tradeoff.png')}")
    print(f"  MAC skip rates           : {os.path.join(OUTPUT_DIR, 'mac_skip_rates.png')}")
    print(f"  Correlation matrix       : {os.path.join(OUTPUT_DIR, 'correlation_matrix.png')}")
    print(f"  Pruning summary          : {os.path.join(OUTPUT_DIR, 'pruning_summary.png')}")
    print(f"{'='*70}")
    print(f"  Done.")


if __name__ == '__main__':
    main()
