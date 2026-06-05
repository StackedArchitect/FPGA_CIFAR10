## ============================================================================
## Timing Constraints — CIFAR-10 CNN (4-Layer, Parallel + BN)
##
## Target:  Xilinx Zynq-7020 (XC7Z020CLG484-1), ZedBoard
## Clock:   40 MHz (25 ns period)
##
## Architecture:
##   Conv1(3→32, pad=1) → BN1 → Pool1
##   Conv2(32→64, pad=1) → BN2 → Pool2
##   Conv3(64→64, pad=1) → BN3 (no pool)
##   Conv4(64→64, pad=1) → BN4 (no pool)
##   GAP(8×8→1×1) → FC1(64→256) → BN5 → FC2(256→10)
##
## 16 parallel multipliers per group, Q16.16 fixed-point.
## All weights/biases/BN loaded internally via $readmemh (BRAM ROMs).
## ============================================================================

## ---- Primary clock ----
create_clock -period 25.000 -name sys_clk [get_ports clk]

## ---- I/O delays ----
set_input_delay  -clock sys_clk 5.000 [get_ports rstn]
set_output_delay -clock sys_clk 5.000 [get_ports pred_out[*]]

## ---- Async reset false path ----
## rstn is asynchronous to sys_clk — no timing analysis needed
set_false_path -from [get_ports rstn]

## ---- Fanout limits ----
## Prevent Vivado from creating massive fanout trees on control signals.
## The FSM state registers drive many parallel paths — limit replication.
set_property MAX_FANOUT 64 [get_nets -hierarchical -filter {NAME =~ "*state*"}]

