# ==============================================================================
# synth_settings.tcl — Apply to existing Vivado project for fast synthesis
#
# Usage: In Vivado Tcl Console, run:
#   source C:/Users/ADMIN/Desktop/FPGA_CIFAR10/cifar10_baseline/hardware/synth_settings.tcl
#
# Then click "Run Synthesis" normally — it will use these optimized settings.
# ==============================================================================

# ---- Apply to current synthesis run ----
set_property strategy Flow_RuntimeOptimized [get_runs synth_1]

# ---- Override specific synthesis options ----
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-flatten_hierarchy none -no_timing_driven -resource_sharing off -shreg_min_size 5} \
    -objects [get_runs synth_1]

set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs synth_1]

# ---- Apply to implementation run too ----
set_property strategy Performance_RefinePlacement [get_runs impl_1]
set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]

# ---- Parallel jobs ----
set_property AUTO_INCREMENTAL_CHECKPOINT 1 [get_runs synth_1]

puts ""
puts "============================================================"
puts "  Synthesis settings applied!"
puts "  Strategy: RuntimeOptimized"
puts "  Hierarchy: NOT flattened (each module synthesized separately)"
puts "  Timing-driven: OFF (40 MHz is easy for Zynq-7020)"
puts "  Resource sharing: OFF"
puts "============================================================"
puts ""
puts "  Now click 'Run Synthesis' in the GUI, or run:"
puts "    launch_runs synth_1 -jobs 4"
puts "    wait_on_run synth_1"
puts "    launch_runs impl_1 -jobs 4"
puts "    wait_on_run impl_1"
puts "============================================================"
