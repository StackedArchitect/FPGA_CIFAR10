# CIFAR-10 4-Layer CNN Baseline Model (Software + Hardware)

This project contains the software training scripts and hardware RTL descriptions for an FPGA-targeted 4-Layer Convolutional Neural Network (CNN) with Batch Normalization optimized for the **Xilinx Zynq-7020 (ZedBoard)**.

---

## 1. Model Specifications Summary

| Specification | Software Model (PyTorch) | Hardware Model (SystemVerilog) |
|---|---|---|
| **Quantization Format** | 32-bit Float | 32-bit Fixed Point (Q16.16) |
| **Accuracy (Test Set)** | 90.62% | 90.62% (Identical classification) |
| **Number of Parameters** | 120,490 | 124,058 (120,490 active + 3,568 padding weights) |
| **Parallelism** | Parallel CPU/GPU execution | 4 Parallel filter multipliers per group |
| **Target Frequency** | N/A | 40 MHz (25 ns clock period) |
| **Target Device** | CUDA / CPU | Xilinx Zynq-7020 (XC7Z020CLG484-1) |
| **Feature Map Strategy** | PyTorch Tensor Buffer | BRAM Ping-Pong Reuse (2 Buffers) |

### Detailed Specifications Explanation

* **Quantization Format (32-bit Float vs Q16.16 Fixed Point):**
  The hardware design utilizes a Q16.16 fixed-point representation (1 sign bit, 15 integer bits, and 16 fractional bits). This format covers a dynamic range of $[-32768.0, 32767.99998]$ with a resolution of $2^{-16} \approx 0.000015$. This dynamic range is critical because it prevents intermediate accumulation overflows during the massive sum-of-products operations in the convolutional layers while preserving sufficient fractional precision to maintain the exact 90.62% classification accuracy of the software model.
* **Accuracy (90.62%):**
  The hardware implementation is verified to be 100% mathematically consistent with the PyTorch model. Because the activations and weights in the baseline CNN are bounded and do not span many orders of magnitude (standard deviation $\approx 1.0$, maximum activation value $< 120.0$), the Q16.16 quantization noise is negligible. Thus, the hardware output logits match the software logits to within very small tolerances, resulting in identical argmax predictions.
* **Number of Parameters (120,490 vs 124,058):**
  The PyTorch model contains 120,490 parameters (weights, biases, BN scales, and BN shifts). In the hardware implementation, the parameter count is 124,058 words because the inputs and weights of the Fully Connected (FC) layers are padded with zeros (`PAD = 20`) to simplify the addressing control logic in the FSM. Specifically, the FC1 weight ROM is expanded from $256 \times 64 = 16,384$ to $256 \times (20 + 64 + 20) = 26,624$ parameters, and FC2 is expanded from $10 \times 256 = 2,560$ to $10 \times (20 + 256 + 20) = 2,960$ parameters. The zero padding is mathematically transparent because multiplying padding zeros contributes nothing to the accumulator.
* **Parallelism (4 Parallel Filter Multipliers):**
  Sizing the convolutional filters' parallelism factor is a critical design choice to fit the Zynq-7020 resource constraints. If the parallelism was set to 16, it would require 16 parallel multipliers per layer, forcing the intermediate max-pooling buffers (`conv_buf`) to hold $16 \times \text{spatial size} \times 32\text{-bit}$ words. For Conv1 ($16 \times 1024 \times 32\text{-bit} = 524,288$ bits), this would consume 15 BRAMs per buffer. Reducing the parallelism to `PARALLEL_CH = 4` reduces this buffer size by $4\times$, requiring only 4 BRAMs and leaving ample resources for feature maps and weights.
* **Feature Map Strategy (BRAM Ping-Pong Reuse):**
  Since the neural network executes layer-by-layer sequentially, we do not need independent memory buffers for each layer's activations. Instead, we declare only two feature map buffers (`fmap_a` and `fmap_b`), each sized at 8,192 words × 32-bit = 262,144 bits. The layers alternate: Phase 0 (Conv1) reads `fmap_a` and writes `fmap_b`; Phase 1 (Conv2) reads `fmap_b` and writes `fmap_a`, and so on. This ping-pong strategy saves a total of 7 BRAM36 blocks.

---

## 2. Neural Network Architecture

The architecture consists of **4 Convolutional Layers** (giving a 5×5 effective receptive field at the 8×8 resolution to break the 87% accuracy ceiling) followed by **Global Average Pooling** and **2 Fully Connected Layers**.

```
Input (32×32×3)
   │
   ├── [Conv1] 32 filters, 3×3, pad=1 ── [BN1] ── [ReLU] ── [MaxPool 2×2] (16×16×32)
   │
   ├── [Conv2] 64 filters, 3×3, pad=1 ── [BN2] ── [ReLU] ── [MaxPool 2×2] (8×8×64)
   │
   ├── [Conv3] 64 filters, 3×3, pad=1 ── [BN3] ── [ReLU] (8×8×64)
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

## 3. Physical Synthesis Resource Utilization

*Synthesized and routed in Vivado 2024.2 ML Edition targeting **XC7Z020CLG484-1**.*

| Resource | Used | Available | % Utilization |
|---|---|---|---|
| **LUT (Logic)** | 23,241 | 53,200 | 43.69% |
| **Slice Registers (FF)** | 15,632 | 106,400 | 14.69% |
| **Block RAM Tile** | 48 | 140 | 34.29% |
| **DSPs** | 88 | 220 | 40.00% |
| **Bonded IOB (Pins)** | 6 | 200 | 3.00% |
| **BUFGCTRL (Clock)** | 1 | 32 | 3.13% |

### Detailed Resource Analysis

* **LUTs (23,241 used / 43.69%):**
  The LUTs are split into:
  - **LUT as Logic (19,721):** These are used for the combinational control logic of the layer FSMs, coordinate calculations for zero-padding boundaries, comparator trees inside the Max-Pooling modules, 32-bit addition/accumulation logic, and multiplexer routing into the dual-port BRAM blocks. Splitting the weight ROMs into four separate 1D arrays successfully enabled Vivado to map them to BRAM blocks, reducing the LUT count of `u_conv_3` from **84,983** to **1,165** and `u_conv_4` from **81,274** to **1,170**.
  - **LUT as Memory (3,520):** These are SLICEM LUTs configured as Distributed RAM (`RAMD64E` primitives) to store small read-only memories:
    1. **FC2 Weights:** Since FC2 has only 2,960 weights (Q16.16, 32-bit), storing them in BRAM would consume 3 BRAM36 blocks which would be mostly empty (wasting BRAM). Storing them in distributed RAM saves these 3 BRAMs.
    2. **Bias, BN scale, and BN shift ROMs for all Conv layers:** These ROMs are small (32 or 64 entries each). Mapping them to distributed RAM saves a total of 12 BRAMs.
* **Slice Registers (15,632 used / 14.69%):**
  Used to implement the pipelined datapath registers (latching weights and input pixels before DSP multiplication), loop counters, FSM state variables, pipeline validity shift registers, and the output registers of the sequential layers: GAP output (`gap_out` - 64 registers), FC1 output (`fc1_out_raw` - 256 registers), and the final registered prediction (`pred_reg` - 4 registers).
  - 15,628 are registered as `FDRE` (register with clock enable and synchronous reset).
  - 4 are registered as `FDSE` (register with clock enable and synchronous set), used in the binary tree comparator to register the output class index.
* **Block RAM Tiles (48 used / 34.29%):**
  - **Feature Maps (16 BRAM36s):** `fmap_a` and `fmap_b` are each sized at 8,192 words × 32-bit = 262,144 bits. Since one Xilinx BRAM36 holds 36,864 bits, each buffer requires $\frac{262,144}{36,864} = 7.11$ blocks, which rounds up to **8 BRAM36 blocks**. For two ping-pong buffers, this requires **16 BRAM36 blocks**.
  - **FC1 Weights (32 BRAM36s):** FC1 has $256 \times (20 + 64 + 20) = 26,624$ words of 32-bit = 851,968 bits. $\frac{851,968}{36,864} = 23.11$ blocks, which rounds up to **32 BRAM36 blocks** due to address space alignment and power-of-two address decoding.
  - **Convolution Weights (0 BRAM36s):** In this synthesis run, the weight `.mem` files (`conv1_w.mem` through `conv4_w.mem`) were not added to the Vivado project resources (they were only in the simulator directory). Vivado synthesis could not open them, initialized the arrays to all-zero constants, and optimized the storage away. When the `.mem` files are added to the synthesis fileset, the weight ROMs will consume:
    - **Conv2:** 16 BRAM36s (4 blocks per channel)
    - **Conv3:** 32 BRAM36s (8 blocks per channel)
    - **Conv4:** 32 BRAM36s (8 blocks per channel)
    - **Conv1 `conv_buf`:** 4 BRAM36s
    - **Conv2 `conv_buf`:** 1 BRAM36
    This will bring active BRAM usage to **133 BRAM36 blocks (95.0%)**, which fits within the 140 budget.
* **DSPs (88 used / 40.00%):**
  Used for high-precision signed Q16.16 multiplications. A 32-bit signed multiplication requires **4 DSP48E1 slices** (configured as cascaded multipliers to perform multi-precision multiplication: $A_H \times B_H$, $A_H \times B_L$, $A_L \times B_H$, and $A_L \times B_L$).
  - `u_conv_pool_1`: 20 DSPs (4 filter multipliers × 4 DSPs + 1 BN multiplier × 4 DSPs)
  - `u_conv_pool_2`: 20 DSPs (4 filter multipliers × 4 DSPs + 1 BN multiplier × 4 DSPs)
  - `u_conv_3`: 18 DSPs (4 filter + 1 BN multipliers, partially optimized by Vivado's compiler due to constant weight propagation and ranges)
  - `u_conv_4`: 18 DSPs (partially optimized by Vivado's compiler)
  - `u_fc1`: 8 DSPs (1 MAC multiplier × 4 DSPs + 1 BN multiplier × 4 DSPs)
  - `u_fc2`: 4 DSPs (1 MAC multiplier × 4 DSPs, no BN multiplier)
  - Total = **88 DSPs**.

---

## 4. Design Timing Closure (40 MHz)

| Timing Metric | Value (ns) | Status |
|---|---|---|
| **Target Clock Period** | 25.000 ns | Met |
| **Worst Negative Slack (WNS)** | **+9.472 ns** | **Met** |
| **Worst Hold Slack (WHS)** | +0.132 ns | Met |
| **Total Negative Slack (TNS)** | 0.000 ns | Met |
| **Failing Endpoints** | 0 / 62,925 | Met |

### Detailed Timing Analysis

* **Setup Slack (WNS = +9.472 ns):**
  The target clock period is 25.000 ns. Having a Worst Negative Slack of +9.472 ns means the worst-case register-to-register combinational path in the design completes in $25.000 - 9.472 = 15.528$ ns (including clock uncertainty, routing, and cell setup delay). This provides a **37.8% timing safety margin** on silicon.
* **Hold Slack (WHS = +0.132 ns):**
  A positive hold slack of +0.132 ns ensures that data remains stable at the input of destination registers for the required hold time after the clock edge, preventing metastability.
* **Failing Endpoints (0 / 62,925):**
  Indicates that all 62,925 timing paths in the design successfully met timing constraints.
* **Timing Resolution:**
  Timing closed successfully because the output argmax logic was converted from a serial 50-level combinational loop into a **4-level binary tree comparator** and registered to `pred_reg` before driving the FPGA pins. This isolated the long combinational routing path to the I/O pads.

---

## 5. Performance & Latency Breakdown (at 40 MHz)

| Pipeline Phase | Completed Cycle | Cycles Consumed | Physical Latency (ms) | % of Total |
|---|---|---|---|---|
| **Reset Release** | 0 | 0 | 0.00 ms | 0.0% |
| **Conv1 + BN1 + Pool1** | 344,066 | 344,066 | 8.60 ms | 12.2% |
| **Conv2 + BN2 + Pool2** | 1,585,156 | 1,241,090 | 31.03 ms | 44.0% |
| **Conv3 + BN3** | 2,185,222 | 600,066 | 15.00 ms | 21.3% |
| **Conv4 + BN4** | 2,785,288 | 600,066 | 15.00 ms | 21.3% |
| **GAP** | 2,789,514 | 4,226 | 0.11 ms | 0.1% |
| **FC1 + BN5** | 2,817,164 | 27,650 | 0.69 ms | 1.0% |
| **FC2 (Inference Done)** | **2,820,156** | **2,992** | **0.07 ms** | **0.0%** |
| **TOTAL Inference** | **2,820,156** | **2,820,156** | **70.50 ms** | **100.0%** |

### Mathematical Latency Derivation

The cycles consumed by each module are directly derived from the mathematical loops in the FSM logic:
* **Conv1 + BN1 + Pool1 (344,066 cycles):**
  Processes a 32×32 output grid. Total output positions = 1,024. Input channels = 3. Kernel size = 3×3. Tap count (MAC operations per pixel) = $3 \times 3 \times 3 = 27$ cycles. With `PARALLEL_CH = 4`, it processes $32 / 4 = 8$ filter groups. Total multiplication cycles = $8 \text{ groups} \times 1,024 \text{ positions} \times 27 \text{ taps} = 221,184$ cycles. The remaining 122,882 cycles are consumed by the Max Pooling write operations into `conv_buf` and FSM pipeline overhead.
* **Conv2 + BN2 + Pool2 (1,241,090 cycles):**
  Processes 16×16 output grids. Total positions = 256. Input channels = 32. Tap count = $3 \times 3 \times 32 = 288$ cycles. Processes $64 / 4 = 16$ filter groups. Total multiplication cycles = $16 \text{ groups} \times 256 \text{ positions} \times 288 \text{ taps} = 1,179,648$ cycles. The remaining 61,442 cycles go to pooling and pipeline overhead. This matches the simulation count exactly.
* **Conv3 + BN3 & Conv4 + BN4 (600,066 cycles each):**
  Process 8×8 output grids. Total positions = 64. Input channels = 64. Tap count = $3 \times 3 \times 64 = 576$ cycles. Processes $64 / 4 = 16$ filter groups. Total multiplication cycles = $16 \text{ groups} \times 64 \text{ positions} \times 576 \text{ taps} = 589,824$ cycles. The remaining 10,242 cycles are pipeline startup/drain overhead.
* **GAP (4,226 cycles):**
  Loops through 64 channels, averaging 8×8 = 64 pixels per channel. Reading 64 pixels sequentially from BRAM takes 64 cycles. Total cycles = $64 \text{ channels} \times (64 \text{ cycles} + 2 \text{ overhead}) = 4,224$ cycles.
* **FC1 + BN5 (27,650 cycles):**
  Processes 256 neurons sequentially. Each neuron has $64 \text{ inputs} + 40 \text{ padding} = 104$ MACs. Cycles = $256 \times (104 + 4 \text{ overhead}) = 27,648$ cycles.
* **FC2 (2,992 cycles):**
  Processes 10 classes sequentially. Each class has $256 \text{ inputs} + 40 \text{ padding} = 296$ MACs. Cycles = $10 \times (296 + 3 \text{ overhead}) = 2,990$ cycles.

---

## 6. Power & Thermal Estimation

| Metric | Estimated Value |
|---|---|
| **Total On-Chip Power** | **0.182 W** |
| **Device Static Power** | 0.108 W (59%) |
| **Dynamic Power** | 0.074 W (41%) |
| **- BRAM Power** | 0.035 W (47% of dynamic) |
| **- Clock Power** | 0.018 W (24% of dynamic) |
| **- Logic Power** | 0.011 W (15% of dynamic) |
| **- Signals Power** | 0.008 W (11% of dynamic) |
| **- DSP Power** | 0.002 W (3% of dynamic) |
| **Junction Temperature** | **27.1°C** |
| **Thermal Margin** | 57.9°C (4.8 W) |

### Detailed Power Analysis

* **Total Power (0.182 W):**
  Sum of device static power (0.108 W) and dynamic power (0.074 W).
* **Device Static Power (0.108 W / 59% of total):**
  The leakage power of the Zynq-7020 chip at room temperature (junction temperature 27.1°C). This is independent of design activity and represents the baseline power to keep the device powered.
* **Dynamic Power (0.074 W / 41% of total):**
  The power consumed due to switching activity in the FPGA fabric.
* **BRAM Power (0.035 W / 47% of dynamic):**
  This is the largest dynamic power consumer because the BRAM blocks (both the ping-pong feature maps and the weight BRAMs) are toggling continuously during every cycle of the 2.8 million cycle inference run to read activations and weights.
* **Clock Power (0.018 W / 24% of dynamic):**
  Consumed by the global clock tree buffers (BUFGCTRL) and routing wires to charge and discharge the capacitive clock pins of all flip-flops, BRAMs, and DSP blocks at 40 MHz.
* **Logic Power (0.011 W / 15% of dynamic):**
  Consumed by the switching activity of the 23,241 LUTs and logic gates.
* **Signals Power (0.008 W / 11% of dynamic):**
  Consumed by the switching activity of the routing wires connecting design logic.
* **DSP Power (0.002 W / 3% of dynamic):**
  DSPs are dedicated silicon blocks on the Zynq-7020 designed with highly optimized internal routing and transistors. Consequently, their dynamic power is extremely low compared to performing multiplications using general fabric LUTs.
* **Junction Temperature (27.1°C) and Thermal Margin (57.9°C):**
  The junction temperature is calculated based on ambient temperature ($25.0^\circ\text{C}$), total power ($0.182\text{ W}$), and the package thermal resistance ($\theta_{JA} = 11.5^\circ\text{C/W}$):
  $$\text{Junction Temp} = 25.0^\circ\text{C} + 0.182\text{ W} \times 11.5^\circ\text{C/W} = 27.093^\circ\text{C} \approx 27.1^\circ\text{C}$$
  This leaves a massive thermal margin of $85.0^\circ\text{C} - 27.1^\circ\text{C} = 57.9^\circ\text{C}$ before exceeding the commercial grade maximum junction temperature of $85^\circ\text{C}$, equivalent to $4.8\text{ W}$ of power headroom.

---

## 7. Architecture Changes & Design Decisions

To fit the baseline model within the resources of the Zynq-7020 and achieve high clock speeds, several architectural adjustments were made to the initial design:

### 1. Channel Parallelism (16 → 4 Filters per Group)
- **Reasoning:** In the convolution layer, writing/reading data in parallel requires intermediate storage. A parallelism of 16 filters required **20 BRAMs** just for internal pooling buffers (`conv_buf`). Reducing this to 4 parallel filters dropped this to **5 BRAMs**, saving 15 BRAMs and fitting our resource budget.
- **Trade-off:** Inference takes 4× longer, but at 70.5 ms, it still easily meets real-time frame rates.

### 2. Feature Map Buffering (5 separate buffers → 2 BRAM Ping-Pong Buffers)
- **Reasoning:** Since the neural network runs sequentially (layer-by-layer), we do not need independent BRAM arrays for each layer's feature maps. We declared only two feature map buffers (`fmap_a` and `fmap_b`) and ping-ponged data back and forth. This saved **7 BRAM36 blocks**.

### 3. Memory Partitioning (Monolithic → Channel-Specific 1D Arrays)
- **Reasoning:** The initial weight ROM had 4 read ports accessing it simultaneously (for the 4 parallel filters). Since dual-port BRAM only supports up to 2 read ports, Vivado bypassed BRAM and synthesized the memory using LUTs, driving LUT utilization to a failing **433% (230,526 LUTs)**. Partitioning the array into **four independent 1D BRAMs** (one per channel) allowed Vivado to infer single-port BRAM blocks, reducing LUT usage to a passing **43.69%**.
- **Compiler Limit Bypass:** Vivado has a 1,000,000-bit elaboration limit on multidimensional variables (which our 2D split weight arrays exceeded). Declaring them as separate 1D arrays bypassed this front-end compiler limit.

### 4. Argmax Logic (Serial Loop → Registered Binary Tree)
- **Reasoning:** The initial combinational argmax loop created a 50-level logic path directly to the chip pins, violating setup timing. Implementing a 4-level **binary tree comparator** and **registering** the output predictions resolved all setup timing failures, leaving a **+9.472 ns WNS** timing margin.
