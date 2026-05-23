# CIFAR-10 CNN — Software Model (Baseline + BatchNorm, v4)

Full-precision 2D CNN trained on CIFAR-10 using **Knowledge Distillation** from a ResNet-18 teacher. Designed for direct deployment to a Zynq-7020 FPGA (XC7Z020-CLG484-1) via Q16.16 fixed-point weight export.

---

## Model Summary

| Property | Value |
|---|---|
| **Test Accuracy** | **90.10%** |
| **Architecture** | 4 Conv + GAP + 2 FC |
| **Parameters** | 113,418 |
| **Training Method** | Knowledge Distillation (KD) |
| **Teacher Model** | ResNet-18 (95.24% accuracy) |
| **Weight Format** | Q16.16 fixed-point (32-bit, two's complement) |
| **Target Hardware** | Zedboard XC7Z020-CLG484-1 @ 40 MHz |

---

## Architecture

```
Input: 32×32×3 (CIFAR-10 image, channel-first)
    │
    ▼
┌─────────────────────────────────────────────┐
│  Conv1: 3→32, 3×3, pad=1                   │
│  BatchNorm2d(32) → ReLU → MaxPool(2×2)     │
│  Output: 16×16×32                           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Conv2: 32→64, 3×3, pad=1                  │
│  BatchNorm2d(64) → ReLU → MaxPool(2×2)     │
│  Output: 8×8×64                             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Conv3: 64→64, 3×3, pad=1  (NO pool)       │
│  BatchNorm2d(64) → ReLU                    │
│  Output: 8×8×64                             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Conv4: 64→64, 3×3, pad=1  (NO pool)       │
│  BatchNorm2d(64) → ReLU                    │
│  Output: 8×8×64                             │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Global Average Pooling: 8×8 → 1×1         │
│  Output: 64                                 │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  FC1: 64→256                                │
│  BatchNorm1d(256) → ReLU → Dropout(0.3)    │
│  Output: 256                                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  FC2: 256→10  (raw logits, no activation)   │
│  Output: 10 (one per CIFAR-10 class)        │
└─────────────────────────────────────────────┘
```

### Why 4 convolutional layers?

Conv3 and Conv4 operate without pooling at 8×8 spatial resolution. Stacking two 3×3 convolutions yields a **5×5 effective receptive field**, which is critical for capturing mid-level spatial features at this resolution. Three-layer variants plateau at ~87-88% due to insufficient depth.

### Layer dimensions

| Layer | Input Size | Output Size | Weights | Parameters |
|-------|-----------|-------------|---------|------------|
| Conv1 | 32×32×3 | 16×16×32 | 3×32×3×3 = 864 | 896 |
| Conv2 | 16×16×32 | 8×8×64 | 32×64×3×3 = 18,432 | 18,496 |
| Conv3 | 8×8×64 | 8×8×64 | 64×64×3×3 = 36,864 | 36,928 |
| Conv4 | 8×8×64 | 8×8×64 | 64×64×3×3 = 36,864 | 36,928 |
| GAP | 8×8×64 | 64 | — | 0 |
| FC1 | 64 | 256 | 64×256 = 16,384 | 16,640 |
| FC2 | 256 | 10 | 256×10 = 2,560 | 2,570 |
| BN (all) | — | — | — | 960 |
| **Total** | | | | **113,418** |

---

## Training Methodology

### Knowledge Distillation (KD)

Instead of training the small student network from hard labels alone, we train it to match the **soft probability distributions** produced by a large, high-accuracy teacher network.

```
                    soft labels
  ┌───────────┐   (e.g. "cat≈82%, dog≈8%,    ┌───────────┐
  │ ResNet-18  │    frog≈3%, ...")            │  Student   │
  │ (teacher)  │ ─────────────────────────►   │  (4-conv)  │
  │  11M params│                              │  113K params│
  │  95.24%    │                              │  90.10%    │
  └───────────┘                              └───────────┘
```

**Why KD works**: Hard labels (`cat = [0,0,0,1,0,0,0,0,0,0]`) discard inter-class similarity information. The teacher's soft predictions (`cat = [0.01, 0.01, 0.02, 0.82, 0.03, 0.08, ...]`) tell the student *which classes look similar to each other*, enabling it to learn richer feature representations than it could from hard labels alone.

### KD Loss Function

```
L = α · CE(student_logits, hard_labels) + (1-α) · T² · KL(student_soft, teacher_soft)
```

Where:
- `CE` = cross-entropy loss (standard classification)
- `KL` = Kullback-Leibler divergence (matches teacher's soft distribution)
- `T = 4.0` = temperature (smooths probability distributions)
- `α = 0.3` = weighting (30% hard labels, 70% teacher guidance)

The temperature `T` softens the output distributions: higher T → flatter distributions → more informative gradients from the teacher's dark knowledge (non-target class relationships).

### Two-Phase Training

| Phase | Model | Optimizer | Schedule | Epochs | Result |
|-------|-------|-----------|----------|--------|--------|
| **Phase 1** | ResNet-18 teacher | SGD (lr=0.1, momentum=0.9, Nesterov) | MultiStepLR([100,150], γ=0.1) | 200 | 95.24% |
| **Phase 2** | 4-conv student | Adam (lr=1e-3, wd=5e-4) | CosineAnnealingLR (T_max=400) | 400 | 90.10% |

- The teacher is cached to `resnet18_teacher_cifar10.pth` after Phase 1. Re-runs skip teacher training entirely.
- The student's best checkpoint (by test accuracy) is saved to `cifar10_cnn_baseline_model.pth`.

### Data Augmentation

Applied **only during training** (not during evaluation or export):

| Augmentation | Purpose |
|---|---|
| `RandomCrop(32, padding=4)` | Translation invariance — shifts the image by up to 4 pixels |
| `RandomHorizontalFlip()` | Horizontal symmetry — CIFAR-10 classes are mostly symmetric |
| `ColorJitter(0.2, 0.2, 0.2)` | Robustness to brightness, contrast, and saturation variations |
| `Cutout(n_holes=1, length=10)` | Occlusion regularization — masks a random 10×10 patch |

**Cutout** was the key regularization technique. Without it, the 4-conv model overfits (93% train, 89% test). With Cutout(10), the generalization gap closes to ~2%, pushing test accuracy above 90%.

### Weight Initialization

- **Conv layers**: Kaiming Normal (He initialization), optimized for ReLU
- **BatchNorm**: γ=1, β=0 (standard identity initialization)
- **FC layers**: Kaiming Normal

### Gradient Clipping

`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` prevents gradient explosions during KD training (the KL term can produce large gradients at high temperatures).

---

## Batch Normalization — Inference Folding

During training, BatchNorm learns running statistics (mean μ, variance σ²) and affine parameters (γ, β). At inference, these are **folded** into a per-channel linear transform:

```
BN(x) = γ · (x - μ) / √(σ² + ε) + β

     = (γ / √(σ² + ε)) · x + (β - μ · γ / √(σ² + ε))
       ^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^
           bn_scale                   bn_shift
```

This means each channel's BN reduces to: `output = bn_scale × input + bn_shift`

The exported `.mem` files contain these precomputed `bn_scale` and `bn_shift` values, allowing the FPGA to apply BN as a single multiply-add per channel — no division or square root needed.

---

## Fixed-Point Export (Q16.16)

All weights, biases, and BN parameters are exported as **Q16.16 fixed-point** values:

```
Q16.16 format:
  Bits [31]     = sign
  Bits [30:16]  = integer part (15 bits + sign = range ±32768)
  Bits [15:0]   = fractional part (precision = 1/65536 ≈ 0.0000153)

Example:  1.5 → 1.5 × 65536 = 98304 → 0x00018000
Example: -0.5 → -0.5 × 65536 = -32768 → 0xFFFF8000 (two's complement)
```

### Exported Files

All files are in the `../weights/` directory, formatted as hex strings (one value per line):

| File | Shape | Entries | Description |
|------|-------|---------|-------------|
| `conv1_w.mem` | 32×3×3×3 | 864 | Conv1 kernel weights |
| `conv1_b.mem` | 32 | 32 | Conv1 biases |
| `conv1_bn_scale.mem` | 32 | 32 | Conv1 BN folded scale (γ/√(σ²+ε)) |
| `conv1_bn_shift.mem` | 32 | 32 | Conv1 BN folded shift (β - μ·scale) |
| `conv2_w.mem` | 64×32×3×3 | 18,432 | Conv2 kernel weights |
| `conv2_b.mem` | 64 | 64 | Conv2 biases |
| `conv2_bn_scale.mem` | 64 | 64 | Conv2 BN folded scale |
| `conv2_bn_shift.mem` | 64 | 64 | Conv2 BN folded shift |
| `conv3_w.mem` | 64×64×3×3 | 36,864 | Conv3 kernel weights |
| `conv3_b.mem` | 64 | 64 | Conv3 biases |
| `conv3_bn_scale.mem` | 64 | 64 | Conv3 BN folded scale |
| `conv3_bn_shift.mem` | 64 | 64 | Conv3 BN folded shift |
| `conv4_w.mem` | 64×64×3×3 | 36,864 | Conv4 kernel weights |
| `conv4_b.mem` | 64 | 64 | Conv4 biases |
| `conv4_bn_scale.mem` | 64 | 64 | Conv4 BN folded scale |
| `conv4_bn_shift.mem` | 64 | 64 | Conv4 BN folded shift |
| `fc1_w.mem` | 256×104 | 26,624 | FC1 weights (with 20-padding on each side) |
| `fc1_b.mem` | 256 | 256 | FC1 biases |
| `fc1_bn_scale.mem` | 256 | 256 | FC1 BN folded scale |
| `fc1_bn_shift.mem` | 256 | 256 | FC1 BN folded shift |
| `fc2_w.mem` | 10×296 | 2,960 | FC2 weights (with 20-padding on each side) |
| `fc2_b.mem` | 10 | 10 | FC2 biases |
| `data_in.mem` | 3×32×32 | 3,072 | Test image (channel-first, normalized) |
| `expected_label.mem` | 1 | 1 | Ground truth label for the test image |

### FC Weight Padding

FC weights are stored with `PAD=20` zero entries on each side of each neuron's weight vector. This aligns with the hardware's sequential MAC addressing scheme, where the layer_seq module reads `[PAD | actual_weights | PAD]` per neuron.

```
FC1 layout per neuron:  [20 zeros] [64 weights] [20 zeros] = 104 entries
FC2 layout per neuron:  [20 zeros] [256 weights] [20 zeros] = 296 entries
```

---

## FPGA Resource Estimate (XC7Z020)

| Resource | Conv1 | Conv2 | Conv3 | Conv4 | FC1 | FC2 | Buffers | **Total** | **Available** |
|----------|-------|-------|-------|-------|-----|-----|---------|-----------|---------------|
| BRAM36 | LUT | 18 | 36 | 36 | 26 | 3 | 14 | **133** | **140** |

Conv1 weights (864 entries) are small enough for distributed RAM (LUT-ROM). All other weight arrays use block RAM (BRAM36).

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib
```

### Training

```bash
python cifar10_cnn_baseline.py
```

- **Phase 1** (teacher): ~5 min on H100, ~30 min on T4. Skipped if `resnet18_teacher_cifar10.pth` exists.
- **Phase 2** (student): ~35 min on H100, ~3 hours on T4.
- Automatically exports `.mem` files to `../weights/` upon completion.

### Output Files

```
software/
├── cifar10_cnn_baseline.py          # Training script
├── cifar10_cnn_baseline_model.pth   # Best student checkpoint (PyTorch)
├── resnet18_teacher_cifar10.pth     # Cached teacher model
├── cifar10_training_curve.png       # Loss/accuracy plots
└── README.md                        # This file

weights/
├── conv[1-4]_w.mem                  # Convolution kernel weights
├── conv[1-4]_b.mem                  # Convolution biases
├── conv[1-4]_bn_scale.mem           # BN folded scale per channel
├── conv[1-4]_bn_shift.mem           # BN folded shift per channel
├── fc[1-2]_w.mem                    # FC weights (padded)
├── fc[1-2]_b.mem                    # FC biases
├── fc1_bn_scale.mem                 # FC1 BN folded scale
├── fc1_bn_shift.mem                 # FC1 BN folded shift
├── data_in.mem                      # Test image
└── expected_label.mem               # Expected class label
```

---

## Training History

| Version | Architecture | Technique | Test Accuracy |
|---------|-------------|-----------|:------------:|
| v1 | 3-conv (32,64,64) | Baseline | 86.68% |
| v2 | 3-conv (64,64,64) | Wider + Cutout(16) | 86.07% |
| v3 | 3-conv (64,64,64) | KD (ResNet-18 teacher) | 87.09% |
| v4a | 4-conv (32,64,64,64) | KD only | 88.93% |
| v4b | 4-conv (32,64,64,64) | KD + Cutout(16) | 89.59% |
| **v4c** | **4-conv (32,64,64,64)** | **KD + Cutout(10)** | **90.10%** |

Key insight: **depth > width** for CNN accuracy on CIFAR-10. Adding a 4th convolutional layer (+1.84%) combined with properly calibrated regularization (+1.17%) broke through the 90% barrier.

---

## CIFAR-10 Classes

| Index | Class |
|:-----:|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

---

## Reproducibility

Results may vary slightly (±0.3%) across runs due to:
- CUDA non-determinism in cuDNN convolution algorithms
- Random initialization seeds
- Stochastic data augmentation (RandomCrop, HFlip, Cutout)

To improve reproducibility, set:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Note: deterministic mode reduces training speed by ~20%.
