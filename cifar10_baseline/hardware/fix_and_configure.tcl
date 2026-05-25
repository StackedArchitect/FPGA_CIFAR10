# ==============================================================================
# fix_and_configure.tcl — Fix duplicates + Set fast synthesis settings
#
# Usage: In Vivado Tcl Console:
#   source C:/Users/ADMIN/Desktop/FPGA_CIFAR10/cifar10_baseline/hardware/fix_and_configure.tcl
#
# Then: Right-click synth_1 → Reset Run → Click "Run Synthesis"
# ==============================================================================

# ==============================================================================
#  STEP 1: Remove duplicate external source files
# ==============================================================================
puts "\n============================================================"
puts "  STEP 1: Removing duplicate external file references"
puts "============================================================"

set external_files_to_remove {}

foreach f [get_files -quiet *.sv] {
    set fpath [get_property NAME $f]
    if {[string match "*FPGA_CIFAR10/cifar10_baseline/hardware*" $fpath] ||
        [string match "*FPGA_CIFAR10\\cifar10_baseline\\hardware*" $fpath]} {
        lappend external_files_to_remove $f
        puts "  Will remove: $fpath"
    }
}

foreach f [get_files -quiet *.xdc] {
    set fpath [get_property NAME $f]
    if {[string match "*FPGA_CIFAR10/cifar10_baseline/hardware*" $fpath] ||
        [string match "*FPGA_CIFAR10\\cifar10_baseline\\hardware*" $fpath]} {
        lappend external_files_to_remove $f
        puts "  Will remove: $fpath"
    }
}

if {[llength $external_files_to_remove] > 0} {
    foreach f $external_files_to_remove {
        remove_files $f
    }
    puts "  Removed [llength $external_files_to_remove] duplicate external files."
} else {
    puts "  No external duplicates found — already clean."
}

# ==============================================================================
#  STEP 2: Set fast synthesis settings
# ==============================================================================
puts "\n============================================================"
puts "  STEP 2: Configuring synthesis for speed"
puts "============================================================"

set_property strategy {Vivado Synthesis Defaults} [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs synth_1]
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} \
    -value {-flatten_hierarchy none -no_timing_driven -resource_sharing off -shreg_min_size 5} \
    -objects [get_runs synth_1]

puts "  Strategy:             Vivado Synthesis Defaults"
puts "  Directive:            RuntimeOptimized"
puts "  Hierarchy flattening: NONE"
puts "  Timing-driven synth:  DISABLED"

# ==============================================================================
#  STEP 3: Set fast implementation settings
# ==============================================================================
puts "\n============================================================"
puts "  STEP 3: Configuring implementation for speed"
puts "============================================================"

set_property STEPS.OPT_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE RuntimeOptimized [get_runs impl_1]

puts "  Implementation: RuntimeOptimized"

# ==============================================================================
#  STEP 4: Verify
# ==============================================================================
puts "\n============================================================"
puts "  STEP 4: Current source files"
puts "============================================================"

puts "\n  Design sources:"
foreach f [get_files -filter {FILE_TYPE == SystemVerilog || FILE_TYPE == Verilog} -of_objects [get_filesets sources_1]] {
    puts "    [get_property NAME $f]"
}

puts "\n  Constraints:"
foreach f [get_files -filter {FILE_TYPE == XDC} -of_objects [get_filesets constrs_1]] {
    puts "    [get_property NAME $f]"
}

puts "\n============================================================"
puts "  DONE! Now:"
puts "    1. Right-click synth_1 -> Reset Run"
puts "    2. Click 'Run Synthesis'"
puts "    3. Expected time: ~15-25 minutes (down from 3+ hours)"
puts "============================================================\n"
