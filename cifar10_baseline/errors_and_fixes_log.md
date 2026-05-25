# Chronological Log of Errors and Fixes

This file documents all the critical warnings, syntax errors, timing violations, and resource over-utilization issues encountered during the development, simulation, and synthesis of the CIFAR-10 CNN baseline model on the Xilinx Zynq-7020 (ZedBoard), along with the hardware and software engineering solutions implemented.

---

## 1. HDL Source Duplication & Top Specification Overwrite

* **Error Messages:**
  - `[HDL 9-3756] overwriting previous definition of module 'cnn2d_synth_top_cifar'`
  - `[filemgmt 20-736] The current top specification, "cnn2d_synth_top_cifar", does not uniquely identify a single design element.`
  - `[filemgmt 20-1318] Duplicate Design Unit 'cnn2d_synth_top_cifar()' found in library 'xil_defaultlib'`

* **Root Cause:**
  The Vivado project database contained duplicate file references pointing to two different directories containing the same source files: the local project directory (`C:/Users/ADMIN/Desktop/cifar10_baseline/...`) and the workspace repository directory (`C:/Users/ADMIN/Desktop/FPGA_CIFAR10/...`). This led to duplicate definitions of key SystemVerilog modules in the synthesis compilation database, causing the tool to fail to identify a unique top module.

* **The Fix:**
  We created and executed a TCL cleanup script [fix_and_configure.tcl](file:///C:/Users/ADMIN/Desktop/FPGA_CIFAR10/cifar10_baseline/hardware/fix_and_configure.tcl) inside Vivado to scan the project filesets and automatically remove duplicate external references, keeping only the local project files.

---

## 2. Synthesis Hang & Out of Memory (3+ Hours)

* **Error/Symptom:**
  Synthesis was running for 3+ hours without completing, exhausting system resources.

* **Root Cause:**
  The initial architecture passed full feature maps between layers as massive unpacked array ports (e.g., `input wire signed [31:0] data_in [0 : 8191]`). In SystemVerilog, passing large unpacked arrays as ports creates individual physical wire lines for every single element. Across all layers, this required **655,360 flip-flops (FFs)** to store intermediate activations. Since the Zynq-7020 FPGA only has **106,400 FFs** in total, Vivado spent hours attempting to route and optimize a design that exceeded the chip's physical limits by **6.2×**.

* **The Fix:**
  We refactored the design to store feature maps in Block RAM (BRAM) arrays located inside the parent module (`cnn2d_top_cifar.sv`), passing indices and values via narrow address/data ports. We also implemented a **BRAM Ping-Pong buffer** strategy, reusing two main BRAMs across the sequential pipeline phases:
  - Phase 0 (Conv1): read BRAM_A (input)  → write BRAM_B (pool1)
  - Phase 1 (Conv2): read BRAM_B (pool1)  → write BRAM_A (pool2)
  - Phase 2 (Conv3): read BRAM_A (pool2)  → write BRAM_B (conv3)
  - Phase 3 (Conv4): read BRAM_B (conv3)  → write BRAM_A (conv4)
  - Phase 4 (GAP):   read BRAM_A (conv4)
  This reduced the flip-flop requirement to just **15,632 (14.69%)** and brought synthesis time down to **~15 minutes**.

---

## 3. Pipeline Weight/Data Misalignment (Simulation Failure)

* **Error/Symptom:**
  The testbench simulation compiled and ran but predicted class `2` (bird) instead of the expected class `3` (cat).

* **Root Cause:**
  During the BRAM memory refactor of the sequential layer module (`layer_seq_cifar.sv`), weight reads were accidentally double-registered:
  - Stage 1: `w_rd_reg <= w_rom_bram[w_addr]`
  - Stage 2: `cur_weight <= w_rom_val` (where `w_rom_val` was the output wire of `w_rd_reg`).
  This introduced an extra clock cycle of latency on the weight path. Because the input feature map data path was only registered once (`cur_data <= data_in[input_idx]`), the weights and data became misaligned by 1 cycle, pairing weights with the wrong inputs during MAC operations.

* **The Fix:**
  We reverted the weight path to match the original single-register datapath by reading from BRAM combinationally and registering once into `cur_weight`:
  ```systemverilog
  always @(posedge clk) begin
      cur_weight <= w_rom_combo; // Read combinationally from ROM block -> register
      cur_data   <= data_in[input_idx];
  end
  ```

---

## 4. Output Logic Timing Violation (-5.898 ns Slack)

* **Error Messages:**
  - `[Vivado 12-4739] set_false_path:No valid object(s) found for ...`
  - `Worst Negative Slack (WNS): -5.898 ns`
  - Failing endpoints: Exactly 4 (matching `pred_out[3:0]`).

* **Root Cause:**
  1. The timing constraints file originally marked weight, bias, and BN ROMs as false paths. Since these ROMs are read-only constants, Vivado's elaboration engine optimized them to hardwired constants (GND/VCC), meaning they had no sequential cells, which triggered the critical warnings. Furthermore, setting a false path on `w_rom` disabled timing analysis on the multipliers, creating a hardware timing hazard.
  2. The argmax logic (which determines the predicted class) was implemented as a combinational `for` loop in `cnn2d_synth_top_cifar.sv`. This was synthesized into a **serial chain of 9 comparators** (each performing a 32-bit signed subtraction). This created **50 levels of logic** between the FC2 registers and the chip's output pins, violating the 25 ns clock period.

* **The Fix:**
  1. We deleted the redundant and hazardous false path constraints from the XDC file.
  2. We replaced the serial argmax loop with a **registered binary tree argmax**:
     - It compares the 10 outputs in a tree structure (`ceil(log2(10)) = 4` levels of comparators), reducing logic levels from 50 to 4.
     - The output is registered to a flip-flop (`pred_reg`) on `clk`, isolating the output pin path and closing timing with a positive slack of **+9.472 ns**.

---

## 5. Elaboration Limit Exceeded (`Synth 8-4556`)

* **Error Message:**
  `[Synth 8-4556] size of variable 'w_rom_split' is too large to handle; the size of the variable is 1179648, the limit is 1000000`

* **Root Cause:**
  To solve the BRAM port contention (which was causing 433% LUT usage), we partitioned the monolithic weights into a 2D array of size `[0 : PARALLEL_CH-1][0 : SUB_W_SIZE-1]`. In Vivado, multidimensional unpacked arrays are treated as a single variable of size `width * depth` bits during elaboration. Since Conv3/4 weights are `4 * 9,216 * 32 = 1,179,648` bits, it exceeded Vivado's built-in elaboration limit of 1,000,000 bits.

* **The Fix:**
  We declared **four separate 1D arrays** (`w_rom_split_0` through `w_rom_split_3`) inside a parameter-safe `generate` block. Since 1D unpacked arrays are parsed differently by Vivado, they are not subject to the 1,000,000-bit multidimensional limit, allowing them to compile cleanly and map to BRAMs.
