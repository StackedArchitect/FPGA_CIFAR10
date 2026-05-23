##============================================================================
## Unified Timing & I/O Constraints — All CNN Designs
## Target: Xilinx Zynq-7020 (XC7Z020CLG484-1)
##
## Clock: 40 MHz (25.000 ns period)
## This single constraint file is used across ALL design variants
## (Baseline Seq/Par, Baseline+BN, TTQ+BN, TTQ+Threshold, TTQ+Hysteresis)
## to enable fair, apples-to-apples comparison of timing, power, and area.
##
## 25 ns chosen to accommodate the longest critical path (~21.3 ns in
## TTQ+Hysteresis pruning) with ~3.7 ns margin.
##============================================================================

## Primary clock — 40 MHz
create_clock -period 25.000 -name clk [get_ports clk]

## Input / output delay constraints
set_input_delay  -clock clk 2.0 [get_ports rstn]
set_output_delay -clock clk 2.0 [get_ports pred_out*]
