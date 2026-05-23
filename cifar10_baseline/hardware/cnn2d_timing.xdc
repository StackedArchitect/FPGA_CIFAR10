## ============================================================================
## Timing Constraints — CIFAR-10 Baseline CNN (Parallel + BN)
##
## Target:  Xilinx Zynq-7020 (XC7Z020CLG484-1), ZedBoard
## Clock:   40 MHz (25 ns period)
##
## Architecture:
##   Conv1(3→32, pad=1) → BN1 → Pool1
##   Conv2(32→64, pad=1) → BN2 → Pool2
##   Conv3(64→64, pad=1) → BN3 (no pool)
##   GAP(8×8→1×1) → FC1(64→256) → BN4 → FC2(256→10)
##
## 16 parallel multipliers per group, Q16.16 fixed-point.
## ============================================================================

## Primary system clock — 40 MHz
create_clock -period 25.000 -name sys_clk [get_ports clk]

## Input delay (assume synchronous — data_in ROM co-located)
set_input_delay -clock sys_clk 5.000 [get_ports rstn]

## Output delay
set_output_delay -clock sys_clk 5.000 [get_ports pred_out[*]]
