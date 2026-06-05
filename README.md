# CIFAR-10 4-Layer CNN FPGA Accelerator (Software + Hardware)

This project contains the software training scripts and hardware RTL descriptions for an FPGA-targeted 4-Layer Convolutional Neural Network (CNN) with Trained Ternary Quantization (TTQ), Batch Normalization (BN), and Activation Pruning optimizations targeting the **Xilinx Zynq-7020 (ZedBoard)**.

The project is structured into three main design phases:
1. **TTQ + BN (Baseline)**: A quantized baseline without activation pruning.
2. **TTQ + BN + Threshold (DAAP)**: An accelerator incorporating Density-Adaptive Activation Pruning.
3. **TTQ + BN + Hysteresis**: An accelerator incorporating spatial hysteresis gating to prune activations.

---

## 1. Architecture Comparison Table

| Architecture Feature | 1. TTQ + BN (Baseline) | 2. TTQ + BN + Threshold (DAAP) | 3. TTQ + BN + Hysteresis |
| :--- | :---: | :---: | :---: |
| **Layer Structure** | Conv1-4 + GAP + FC1-2 | Conv1-4 + GAP + FC1-2 | Conv1-4 + GAP + FC1-2 |
| **Quantization** | 2-bit Ternary Weights ($W_p$/$W_n$ scales), Q16.16 Activations | 2-bit Ternary Weights ($W_p$/$W_n$ scales), Q16.16 Activations | 2-bit Ternary Weights ($W_p$/$W_n$ scales), Q16.16 Activations |
| **Pruning Strategy** | None (All activations processed) | Density-Adaptive Activation Pruning (DAAP) | Spatial Hysteresis + DAAP |
| **Gating Criterion** | None | $\|x\| < \tau$ (Magnitude-only thresholding) | $\|x\| < \tau$ unless spatial neighbors are active |
| **Mask Storage** | None | Distributed LUTRAM (0-cycle read) | Distributed LUTRAM + Status BRAM line buffers |
| **Mask Resolution** | None | Per-filter/neuron thresholds | Per-filter/neuron thresholds + neighbor context |
| **Hardware Overhead** | Baseline footprint | Comparators + sequential LUTRAM | Comparators + 1.5 BRAM tiles + spatial resolvers |

---

## 2. Project Specifications & Performance Summary

| Specification | Float Baseline (PyTorch) | TTQ + BN Baseline (RTL) | TTQ + BN + Threshold (RTL) | TTQ + BN + Hysteresis (RTL) |
|---|:---:|:---:|:---:|:---:|
| **Quantization Format** | 32-bit Float | 2-bit weights, Q16.16 | 2-bit weights, Q16.16 | 2-bit weights, Q16.16 |
| **Number of Parameters** | 120,490 | 124,058 | 124,058 | 124,058 |
| **Total Inference Cycles** | N/A | **2,877,770** | **2,017,961** | **1,475,264** |
| **Inference Time (@ 40 MHz)**| N/A | **71.94 ms** | **50.45 ms** | **36.88 ms** |
| **Throughput (FPS)** | N/A | **13.90 FPS** | **19.82 FPS** | **27.11 FPS** |
| **Inference Latency Reduction**| Reference | Reference | **29.88% (1.43× speedup)**| **48.74% (1.95× speedup)**|
| **Slice LUTs (XC7Z020)** | N/A | **24,689 (46.41%)** | **31,828 (59.83%)** | **35,798 (67.29%)** |
| **Block RAM Tiles (36K)** | N/A | **16.0 (11.43%)** | **16.0 (11.43%)** | **17.50 (12.50%)** |
| **DSP Slices (XC7Z020)** | N/A | **36 (16.36%)** | **36 (16.36%)** | **36 (16.36%)** |
| **On-Chip Power** | N/A | **0.154 W** | **0.162 W** | **0.160 W** |
| **Worst Negative Slack (WNS)**| N/A | **+10.521 ns** | **+10.315 ns** | **+6.225 ns** |

> [!NOTE]
> All hardware designs are timing-closed at the target frequency of **40 MHz** (25.0 ns period) with positive setup and hold slack. Power dissipation is extremely low (~160 mW), making the designs highly suitable for low-power edge AI deployments.

---

## 3. Neural Network Architecture

The network consists of **4 Convolutional Layers** (providing a $5\times 5$ effective receptive field at the $8\times 8$ resolution to break the 3-layer accuracy ceiling) followed by **Global Average Pooling** and **2 Fully Connected Layers**.

```
Input Image (32×32×3)
   │
   ├── [Conv1] 32 filters, 3×3, pad=1 ── [BN1] ── [ReLU] ── [MaxPool 2×2] (16×16×32)
   │     └── Activation Boundary 1 (Generates Mask 1)
   │
   ├── [Conv2] 64 filters, 3×3, pad=1 ── [BN2] ── [ReLU] ── [MaxPool 2×2] (8×8×64)
   │     └── Activation Boundary 2 (Generates Mask 2)
   │
   ├── [Conv3] 64 filters, 3×3, pad=1 ── [BN3] ── [ReLU] (8×8×64)
   │     └── Activation Boundary 3 (Generates Mask 3)
   │
   ├── [Conv4] 64 filters, 3×3, pad=1 ── [BN4] ── [ReLU] (8×8×64)
   │
   ├── [GAP] Global Average Pool (1×1×64)
   │
   ├── [FC1] 256 neurons ── [BN5] ── [ReLU] (256)
   │
   └── [FC2] 10 neurons (10 raw logits output)
```

---

## 4. Physical Synthesis Resource Analysis

### Slice LUTs (XC7Z020: 53,200 available)
- **Baseline Design (24,689 LUTs / 46.41%)**: Consumes LUTs primarily for FSM state machine control, coordinate calculations for zero padding, max pooling comparators, and distributed memories. The FC2 weight ROM (2,960 entries) and Biases/BatchNorm scales are mapped to Distributed RAM (`RAMD64E` primitives) to preserve BRAM blocks.
- **Threshold Design (31,828 LUTs / 59.83%)**: Fits perfectly as a physical middle-ground between baseline and Hysteresis. Incorporates magnitude-only DAAP activation threshold comparison logic and localized sequential LUTRAM mask storage inside the convolution and fully connected modules. By using a resource-optimized FSM skip gating strategy, it reduces ROM read ports and routing complexity to save 3,798 LUTs compared to the full 4-tap logic.
- **Hysteresis Design (35,798 LUTs / 67.29%)**: Utilizes additional LUTs to implement the FSMs for three `act_mask_gen_hyst` spatial resolvers (4-cardinal neighbor voting) and the LUTRAM mask storage inside the convolution modules.
- **The v5 LUTRAM Optimization**: The initial hysteresis mask generator used flat register outputs (`mask_out` sized at 8,192 or 4,096 bits), requiring a massive 13-level multiplexer tree (~8,000 LUTs per read) in the consumer convolution layers. This led to a **224% LUT utilization failure (119K LUTs)**. The **v5 LUTRAM optimization** replaced the flat output with write-only ports (`mask_wr_en`, `mask_wr_addr`, `mask_wr_data`) and stored the masks in localized distributed RAMs inside the convolution modules. This cut the LUT utilization of `u_mask_gen_1` from **59,520 LUTs** to **198 LUTs** — a **99.6% reduction** — while preserving the skip logic.

### Block RAM Tiles (XC7Z020: 140 available)
- **Feature Map Buffering (16 BRAM36s)**: Both `fmap_a` and `fmap_b` are sized at 8,192 words × 32-bit = 262,144 bits. Alternating sequential layers write and read from these buffers in a ping-pong fashion, saving 7 BRAMs.
- **FC1 Weights (32 BRAM36s)**: FC1 has 26,624 words of 32-bit (851,968 bits) stored in BRAMs.
- **Hysteresis Status RAMs (1.5 BRAM36s)**: Each of the 3 Hysteresis mask generators instantiates a simple dual-port status memory (BRAM18K block, equivalent to 0.5 BRAM36 tile) to resolve neighbors. The Threshold design has no status RAMs and preserves the baseline BRAM consumption of 16.0 tiles.

### DSP Slices (XC7Z020: 220 available)
- All designs use exactly **36 DSPs**. 
- In standard CNN accelerators, DSP requirements scale with tap multiplication. Under Trained Ternary Quantization (TTQ), the multiplication logic inside the convolution layers is completely replaced by additions into split positive/negative accumulators.
- Multiplications are performed only once per output pixel during BatchNorm scaling and weight scaling ($W_p$ / $W_n$ multiplication). This keeps DSP utilization low and constant across all pruning configurations.

---

## 5. Detailed Design Analysis Report

A detailed design report discussing the latency and hardware utilization deltas, as well as the exact theoretical reasoning for these differences, is tracked in the repository:
👉 **[comparison_analysis_report.md](comparison_analysis_report.md)**

---

## 6. Verification & Running Simulation

To run the SystemVerilog behavioral testbench inside Vivado:
1. Ensure Vivado bin tools are in your PATH.
2. Navigate to the hardware directory of the desired model (e.g., `cifar10_ttq-bn-threshold/hardware`).
3. Run the simulation using `xvlog`, `xelab`, and `xsim`:
   ```bash
   # Compile SV files
   xvlog -sv act_mask_gen_thresh.sv conv_pool_2d_cifar_ttq.sv cnn2d_top_cifar_ttq.sv layer_seq_cifar_ttq.sv global_avg_pool_cifar.sv tb_cnn2d_cifar_ttq.sv
   
   # Elaborate design
   xelab -debug typical tb_cnn2d_cifar_ttq -s sim_ttq
   
   # Run simulation
   xsim sim_ttq -runall
   ```
4. Verify that the simulation finishes with `*** PASS - Prediction matches expected label! ***` and prints the final predicted class and inference cycles.
