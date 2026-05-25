# ==============================================================================
# run_synth_fast.tcl — Optimized Vivado Non-Project Synthesis + Implementation
#
# Usage (from Vivado Tcl Console or batch):
#   vivado -mode batch -source run_synth_fast.tcl
#   OR inside Vivado:  source run_synth_fast.tcl
#
# Target: XC7Z020CLG484-1 (ZedBoard) @ 40 MHz
# Design: cnn2d_synth_top_cifar (CIFAR-10 4-layer CNN)
# ==============================================================================

# ---- Configuration ----
set PART        xc7z020clg484-1
set TOP_MODULE  cnn2d_synth_top_cifar
set NUM_JOBS    4          ;# Parallel threads (adjust to your CPU cores)

# ---- Paths (adjust if needed) ----
set HW_DIR      [file normalize "C:/Users/ADMIN/Desktop/FPGA_CIFAR10/cifar10_baseline/hardware"]
set OUT_DIR     [file normalize "C:/Users/ADMIN/Desktop/FPGA_CIFAR10/cifar10_baseline/output"]

# Create output directory
file mkdir $OUT_DIR

# ---- Source files ----
set SRC_FILES [list \
    "$HW_DIR/conv_pool_2d_cifar.sv" \
    "$HW_DIR/global_avg_pool_cifar.sv" \
    "$HW_DIR/layer_seq_cifar.sv" \
    "$HW_DIR/cnn2d_top_cifar.sv" \
    "$HW_DIR/cnn2d_synth_top_cifar.sv" \
]

# ---- XDC constraints file ----
set XDC_FILE "$HW_DIR/cnn2d_timing.xdc"

# ==============================================================================
#  STEP 1: Read design
# ==============================================================================
puts "============================================================"
puts "  STEP 1: Reading design files"
puts "============================================================"

foreach f $SRC_FILES {
    puts "  Reading: $f"
    read_verilog -sv $f
}
read_xdc $XDC_FILE

# ==============================================================================
#  STEP 2: Synthesis — SPEED-OPTIMIZED SETTINGS
# ==============================================================================
puts "\n============================================================"
puts "  STEP 2: Running Synthesis (RuntimeOptimized)"
puts "============================================================"

# Key synthesis settings for speed:
#
#   -flatten_hierarchy none
#       THE BIGGEST speedup. Prevents Vivado from merging all modules
#       into one giant netlist. Each module is synthesized independently.
#       This alone can reduce synthesis from 3 hours to 20 minutes.
#
#   -directive RuntimeOptimized
#       Uses the fastest synthesis algorithms. Trades off minor QoR
#       for significant runtime reduction (~2-3x faster than default).
#
#   -no_timing_driven
#       Skips timing-driven synthesis optimizations. Since we have a
#       relaxed 40 MHz target on Zynq-7020, timing will be met easily
#       without this expensive optimization pass.
#
#   -shreg_min_size 5
#       Prevents small shift registers from being collapsed into SRLs,
#       which can cause long synthesis exploration.
#
#   -resource_sharing off
#       Prevents resource sharing analysis (which is slow and not needed
#       since each conv module has independent compute paths).

synth_design \
    -top $TOP_MODULE \
    -part $PART \
    -flatten_hierarchy none \
    -directive RuntimeOptimized \
    -no_timing_driven \
    -shreg_min_size 5 \
    -resource_sharing off

# Save checkpoint after synthesis
write_checkpoint -force "$OUT_DIR/post_synth.dcp"

# Generate utilization report
report_utilization -file "$OUT_DIR/utilization_synth.rpt"
report_timing_summary -file "$OUT_DIR/timing_synth.rpt"

puts "\n[INFO] Synthesis complete!"
puts "[INFO] Utilization report: $OUT_DIR/utilization_synth.rpt"

# ==============================================================================
#  STEP 3: Implementation (Opt + Place + Route)
# ==============================================================================
puts "\n============================================================"
puts "  STEP 3: Running Implementation"
puts "============================================================"

# Optimize — light-touch only
opt_design -directive RuntimeOptimized

# Place
place_design -directive RuntimeOptimized

# Physical optimization (post-place)
phys_opt_design -directive RuntimeOptimized

# Route
route_design -directive RuntimeOptimized

# Save checkpoint after implementation
write_checkpoint -force "$OUT_DIR/post_impl.dcp"

# ==============================================================================
#  STEP 4: Reports
# ==============================================================================
puts "\n============================================================"
puts "  STEP 4: Generating Reports"
puts "============================================================"

report_utilization -file "$OUT_DIR/utilization_impl.rpt"
report_timing_summary -file "$OUT_DIR/timing_impl.rpt"
report_power -file "$OUT_DIR/power_impl.rpt"
report_drc -file "$OUT_DIR/drc_impl.rpt"

# ==============================================================================
#  STEP 5: Generate bitstream
# ==============================================================================
puts "\n============================================================"
puts "  STEP 5: Generating Bitstream"
puts "============================================================"

write_bitstream -force "$OUT_DIR/cnn2d_cifar10.bit"

puts "\n============================================================"
puts "  DONE — All files in: $OUT_DIR"
puts "============================================================"
puts "  Bitstream:    $OUT_DIR/cnn2d_cifar10.bit"
puts "  Utilization:  $OUT_DIR/utilization_impl.rpt"
puts "  Timing:       $OUT_DIR/timing_impl.rpt"
puts "============================================================"
