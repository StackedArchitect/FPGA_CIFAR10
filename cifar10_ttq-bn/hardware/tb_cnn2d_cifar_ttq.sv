`timescale 1ns / 1ps
//============================================================================
// Testbench — CIFAR-10 CNN (TTQ + BatchNorm)
//
// Behavioural simulation testbench for cnn2d_synth_top_cifar_ttq.
// Identical structure to baseline testbench (tb_cnn2d_cifar.sv):
//   • 40 MHz clock (25 ns period)
//   • Assert reset for 200 ns, then release
//   • Monitor per-layer done signals
//   • Compare prediction against expected label
//
// To run:
//   cd <project>/cifar10_ttq-bn/hardware/
//   xvlog --sv tb_cnn2d_cifar_ttq.sv cnn2d_synth_top_cifar_ttq.sv \
//          cnn2d_top_cifar_ttq.sv conv_pool_2d_cifar_ttq.sv \
//          layer_seq_cifar_ttq.sv ../cifar10_baseline/hardware/global_avg_pool_cifar.sv
//   xelab tb_cnn2d_cifar_ttq -s sim_ttq
//   xsim sim_ttq -R
//============================================================================
module tb_cnn2d_cifar_ttq;

    // ================================================================
    //  Parameters
    // ================================================================
    localparam INPUT_H      = 32;
    localparam INPUT_W      = 32;
    localparam INPUT_CH     = 3;
    localparam CONV1_OUT_CH = 32;
    localparam CONV2_OUT_CH = 64;
    localparam CONV3_OUT_CH = 64;
    localparam CONV4_OUT_CH = 64;
    localparam FC1_OUT      = 256;
    localparam FC2_OUT      = 10;
    localparam PARALLEL_CH  = 4;
    localparam BITS         = 31;

    // ================================================================
    //  Clock + reset
    // ================================================================
    reg  clk  = 0;
    reg  rstn = 0;
    wire [3:0] pred_out;

    always #12.5 clk = ~clk;  // 40 MHz

    initial begin
        rstn = 0;
        #200;
        rstn = 1;
    end

    // ================================================================
    //  DUT — TTQ synth wrapper
    // ================================================================
    cnn2d_synth_top_cifar_ttq #(
        .INPUT_H     (INPUT_H),
        .INPUT_W     (INPUT_W),
        .INPUT_CH    (INPUT_CH),
        .CONV1_OUT_CH(CONV1_OUT_CH),
        .CONV2_OUT_CH(CONV2_OUT_CH),
        .CONV3_OUT_CH(CONV3_OUT_CH),
        .CONV4_OUT_CH(CONV4_OUT_CH),
        .FC1_OUT     (FC1_OUT),
        .FC2_OUT     (FC2_OUT),
        .PARALLEL_CH (PARALLEL_CH),
        .BITS        (BITS)
    ) u_dut (
        .clk      (clk),
        .rstn     (rstn),
        .pred_out (pred_out)
    );

    // ================================================================
    //  Expected label
    // ================================================================
    reg [31:0] expected_label_mem [0:0];
    initial $readmemh("expected_label.mem", expected_label_mem);
    wire [3:0] expected_label = expected_label_mem[0][3:0];

    // ================================================================
    //  Cycle counter
    // ================================================================
    integer cycle_count = 0;
    always @(posedge clk) begin
        if (rstn)
            cycle_count <= cycle_count + 1;
    end

    // ================================================================
    //  Monitor internal done signals
    //  Access DUT hierarchy: u_dut → u_dut (synth→top) → u_conv_pool_1, etc.
    // ================================================================

    // Conv1+BN1+Pool1
    reg pool1_seen = 0;
    always @(posedge clk) begin
        if (rstn && !pool1_seen && u_dut.u_dut.u_conv_pool_1.done) begin
            pool1_seen <= 1;
            $display("[INFO] Conv1+BN1+Pool1 DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // Conv2+BN2+Pool2
    reg pool2_seen = 0;
    always @(posedge clk) begin
        if (rstn && !pool2_seen && u_dut.u_dut.u_conv_pool_2.done) begin
            pool2_seen <= 1;
            $display("[INFO] Conv2+BN2+Pool2 DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // Conv3+BN3
    reg conv3_seen = 0;
    always @(posedge clk) begin
        if (rstn && !conv3_seen && u_dut.u_dut.u_conv_3.done) begin
            conv3_seen <= 1;
            $display("[INFO] Conv3+BN3 DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // Conv4+BN4
    reg conv4_seen = 0;
    always @(posedge clk) begin
        if (rstn && !conv4_seen && u_dut.u_dut.u_conv_4.done) begin
            conv4_seen <= 1;
            $display("[INFO] Conv4+BN4 DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // GAP
    reg gap_seen = 0;
    always @(posedge clk) begin
        if (rstn && !gap_seen && u_dut.u_dut.u_gap.done) begin
            gap_seen <= 1;
            $display("[INFO] GAP DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // FC1+BN5
    reg fc1_seen = 0;
    always @(posedge clk) begin
        if (rstn && !fc1_seen && u_dut.u_dut.u_fc1.counter_donestatus) begin
            fc1_seen <= 1;
            $display("[INFO] FC1+BN5 DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // FC2 (final layer)
    reg fc2_seen = 0;
    always @(posedge clk) begin
        if (rstn && !fc2_seen && u_dut.u_dut.u_fc2.counter_donestatus) begin
            fc2_seen <= 1;
            $display("[INFO] FC2 DONE at %0t ns (cycle %0d).", $time, cycle_count);
        end
    end

    // ================================================================
    //  Final check — wait for FC2 to complete + 2 settle cycles
    // ================================================================
    reg [2:0] finish_cnt = 0;
    always @(posedge clk) begin
        if (fc2_seen && finish_cnt < 3'd4)
            finish_cnt <= finish_cnt + 1;
    end

    always @(posedge clk) begin
        if (finish_cnt == 3'd3) begin
            $display("");
            $display("############################################################");
            $display("#  CIFAR-10 CNN + TTQ + BN INFERENCE COMPLETE - RESULTS     ");
            $display("############################################################");
            $display("");

            $display("  Predicted class : %0d", pred_out);
            $display("  Expected class  : %0d", expected_label);
            $display("");

            // Print all 10 logits
            $display("  Raw logits (Q16.16 fixed-point):");
            $display("  Class 0: %0d", $signed(u_dut.cnn_out[0]));
            $display("  Class 1: %0d", $signed(u_dut.cnn_out[1]));
            $display("  Class 2: %0d", $signed(u_dut.cnn_out[2]));
            $display("  Class 3: %0d", $signed(u_dut.cnn_out[3]));
            $display("  Class 4: %0d", $signed(u_dut.cnn_out[4]));
            $display("  Class 5: %0d", $signed(u_dut.cnn_out[5]));
            $display("  Class 6: %0d", $signed(u_dut.cnn_out[6]));
            $display("  Class 7: %0d", $signed(u_dut.cnn_out[7]));
            $display("  Class 8: %0d", $signed(u_dut.cnn_out[8]));
            $display("  Class 9: %0d", $signed(u_dut.cnn_out[9]));
            $display("");

            $display("  Total inference cycles: %0d", cycle_count);
            $display("  Total inference time  : %0.3f ms (@ 40 MHz)",
                     cycle_count * 25.0 / 1_000_000.0);
            $display("");

            if (pred_out == expected_label) begin
                $display("  *** PASS — Prediction matches expected label! ***");
            end else begin
                $display("  *** FAIL — Prediction does NOT match expected! ***");
            end

            $display("");
            $display("############################################################");
            $finish;
        end
    end

    // ================================================================
    //  Banner
    // ================================================================
    initial begin
        $display("");
        $display("============================================================");
        $display("  CIFAR-10 CNN + TTQ + BATCHNORM TESTBENCH");
        $display("  Architecture: Conv1(3->32) + Conv2(32->64) + Conv3(64->64)");
        $display("                + Conv4(64->64) + GAP + FC1(64->256) + FC2(256->10)");
        $display("  Quantization: TTQ (2-bit ternary + Wp/Wn scalars)");
        $display("  Parallelism:  %0d filters per group", PARALLEL_CH);
        $display("============================================================");
        $display("");
    end

    // ================================================================
    //  Timeout safety — 200 ms of simulation time = 200,000,000 ns
    // ================================================================
    initial begin
        #200_000_000;  // 200 ms
        $display("[ERROR] Simulation timed out after 200 ms!");
        $display("[DEBUG] Conv1 state = %0d, group_idx = %0d, conv_pos = %0d",
                 u_dut.u_dut.u_conv_pool_1.state,
                 u_dut.u_dut.u_conv_pool_1.group_idx,
                 u_dut.u_dut.u_conv_pool_1.conv_pos);
        $display("[DEBUG] Conv1 done = %0b, phase = %0d",
                 u_dut.u_dut.u_conv_pool_1.done,
                 u_dut.u_dut.phase);
        $finish;
    end

    // ================================================================
    //  Start banner after reset
    // ================================================================
    initial begin
        @(posedge rstn);
        $display("[INFO] Loading expected label (expected_label.mem) ...");
        $display("[INFO] Expected label: %0d", expected_label);
        $display("");
        $display("[INFO] Reset released at %0t ns (cycle 0). Inference running ...", $time);
        $display("");

        // Debug: verify weight/data loading
        $display("[DEBUG] Conv1 w_rom_flat[0] = %0d (2-bit signed)", $signed(u_dut.u_dut.u_conv_pool_1.gen_par_4.w_rom_flat[0]));
        $display("[DEBUG] Conv1 w_rom_flat[1] = %0d (2-bit signed)", $signed(u_dut.u_dut.u_conv_pool_1.gen_par_4.w_rom_flat[1]));
        $display("[DEBUG] Conv1 wp_val = %0d (Q16.16)", u_dut.u_dut.u_conv_pool_1.wp_val);
        $display("[DEBUG] Conv1 wn_val = %0d (Q16.16)", u_dut.u_dut.u_conv_pool_1.wn_val);
        $display("[DEBUG] fmap_a[0] = %0d", $signed(u_dut.u_dut.fmap_a[0]));
        $display("[DEBUG] fmap_a[1] = %0d", $signed(u_dut.u_dut.fmap_a[1]));
        $display("");
    end

    // ================================================================
    //  Periodic progress monitor — Conv1
    // ================================================================
    always @(posedge clk) begin
        if (rstn && !pool1_seen && (cycle_count % 100000 == 0) && cycle_count > 0)
            $display("[PROGRESS] cycle=%0d, Conv1: state=%0d group=%0d pos=%0d",
                     cycle_count,
                     u_dut.u_dut.u_conv_pool_1.state,
                     u_dut.u_dut.u_conv_pool_1.group_idx,
                     u_dut.u_dut.u_conv_pool_1.conv_pos);
    end

endmodule

