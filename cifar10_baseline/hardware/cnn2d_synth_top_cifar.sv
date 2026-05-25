`timescale 1ns / 1ps
//============================================================================
// Synthesis wrapper — CIFAR-10 CNN
//
// Provides file path parameters for all weight/BN/data files.
// The DUT (cnn2d_top_cifar) manages all BRAMs internally.
//
// PARALLEL_CH = 4 — fits BRAM budget (137/140 BRAM36 on XC7Z020)
//
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
module cnn2d_synth_top_cifar #(
    parameter INPUT_H      = 32,
    parameter INPUT_W      = 32,
    parameter INPUT_CH     = 3,
    parameter CONV1_OUT_CH = 32,
    parameter CONV2_OUT_CH = 64,
    parameter CONV3_OUT_CH = 64,
    parameter CONV4_OUT_CH = 64,
    parameter FC1_OUT      = 256,
    parameter FC2_OUT      = 10,
    parameter PARALLEL_CH  = 4,
    parameter BITS         = 31
)(
    input  wire clk,
    input  wire rstn,
    output wire [3:0] pred_out
);

    // ================================================================
    //  DUT output
    // ================================================================
    wire signed [BITS+16:0] cnn_out [0 : FC2_OUT - 1];

    // ================================================================
    //  DUT — all weights, BN, and input data loaded internally via BRAM
    // ================================================================
    cnn2d_top_cifar #(
        .INPUT_H          (INPUT_H),
        .INPUT_W          (INPUT_W),
        .INPUT_CH         (INPUT_CH),
        .CONV1_OUT_CH     (CONV1_OUT_CH),
        .CONV2_OUT_CH     (CONV2_OUT_CH),
        .CONV3_OUT_CH     (CONV3_OUT_CH),
        .CONV4_OUT_CH     (CONV4_OUT_CH),
        .FC1_OUT          (FC1_OUT),
        .FC2_OUT          (FC2_OUT),
        .PARALLEL_CH      (PARALLEL_CH),
        .BITS             (BITS),

        // Conv weight/BN files
        .CONV1_WEIGHT_FILE  ("conv1_w.mem"),
        .CONV1_BIAS_FILE    ("conv1_b.mem"),
        .CONV1_BN_SCALE_FILE("conv1_bn_scale.mem"),
        .CONV1_BN_SHIFT_FILE("conv1_bn_shift.mem"),
        .CONV2_WEIGHT_FILE  ("conv2_w.mem"),
        .CONV2_BIAS_FILE    ("conv2_b.mem"),
        .CONV2_BN_SCALE_FILE("conv2_bn_scale.mem"),
        .CONV2_BN_SHIFT_FILE("conv2_bn_shift.mem"),
        .CONV3_WEIGHT_FILE  ("conv3_w.mem"),
        .CONV3_BIAS_FILE    ("conv3_b.mem"),
        .CONV3_BN_SCALE_FILE("conv3_bn_scale.mem"),
        .CONV3_BN_SHIFT_FILE("conv3_bn_shift.mem"),
        .CONV4_WEIGHT_FILE  ("conv4_w.mem"),
        .CONV4_BIAS_FILE    ("conv4_b.mem"),
        .CONV4_BN_SCALE_FILE("conv4_bn_scale.mem"),
        .CONV4_BN_SHIFT_FILE("conv4_bn_shift.mem"),

        // FC weight/BN files
        .FC1_WEIGHT_FILE    ("fc1_w.mem"),
        .FC2_WEIGHT_FILE    ("fc2_w.mem"),
        .FC1_BIAS_FILE      ("fc1_b.mem"),
        .FC2_BIAS_FILE      ("fc2_b.mem"),
        .FC1_BN_SCALE_FILE  ("fc1_bn_scale.mem"),
        .FC1_BN_SHIFT_FILE  ("fc1_bn_shift.mem"),

        // Input image
        .DATA_IN_FILE       ("data_in.mem")
    ) u_dut (
        .clk     (clk),
        .rstn    (rstn),
        .cnn_out (cnn_out)
    );

    // ================================================================
    //  Binary Tree Argmax (Registered)
    //  Reduces logic levels from 50 (serial chain) to 4 (binary tree)
    //  and registers the output to ensure timing closure at 40 MHz.
    // ================================================================
    
    // Level 1 Comparison Nodes (5 comparators)
    wire signed [BITS+16:0] val_1_0 = (cnn_out[0] >= cnn_out[1]) ? cnn_out[0] : cnn_out[1];
    wire [3:0]              idx_1_0 = (cnn_out[0] >= cnn_out[1]) ? 4'd0 : 4'd1;
    
    wire signed [BITS+16:0] val_1_1 = (cnn_out[2] >= cnn_out[3]) ? cnn_out[2] : cnn_out[3];
    wire [3:0]              idx_1_1 = (cnn_out[2] >= cnn_out[3]) ? 4'd2 : 4'd3;
    
    wire signed [BITS+16:0] val_1_2 = (cnn_out[4] >= cnn_out[5]) ? cnn_out[4] : cnn_out[5];
    wire [3:0]              idx_1_2 = (cnn_out[4] >= cnn_out[5]) ? 4'd4 : 4'd5;
    
    wire signed [BITS+16:0] val_1_3 = (cnn_out[6] >= cnn_out[7]) ? cnn_out[6] : cnn_out[7];
    wire [3:0]              idx_1_3 = (cnn_out[6] >= cnn_out[7]) ? 4'd6 : 4'd7;
    
    wire signed [BITS+16:0] val_1_4 = (cnn_out[8] >= cnn_out[9]) ? cnn_out[8] : cnn_out[9];
    wire [3:0]              idx_1_4 = (cnn_out[8] >= cnn_out[9]) ? 4'd8 : 4'd9;

    // Level 2 Comparison Nodes (2 comparators, 1 pass-through)
    wire signed [BITS+16:0] val_2_0 = (val_1_0 >= val_1_1) ? val_1_0 : val_1_1;
    wire [3:0]              idx_2_0 = (val_1_0 >= val_1_1) ? idx_1_0 : idx_1_1;
    
    wire signed [BITS+16:0] val_2_1 = (val_1_2 >= val_1_3) ? val_1_2 : val_1_3;
    wire [3:0]              idx_2_1 = (val_1_2 >= val_1_3) ? idx_1_2 : idx_1_3;
    
    wire signed [BITS+16:0] val_2_2 = val_1_4;
    wire [3:0]              idx_2_2 = idx_1_4;

    // Level 3 Comparison Nodes (1 comparator, 1 pass-through)
    wire signed [BITS+16:0] val_3_0 = (val_2_0 >= val_2_1) ? val_2_0 : val_2_1;
    wire [3:0]              idx_3_0 = (val_2_0 >= val_2_1) ? idx_2_0 : idx_2_1;
    
    wire signed [BITS+16:0] val_3_1 = val_2_2;
    wire [3:0]              idx_3_1 = idx_2_2;

    // Level 4 Comparison Node (1 comparator)
    wire [3:0] next_pred = (val_3_0 >= val_3_1) ? idx_3_0 : idx_3_1;

    // Output Register
    reg [3:0] pred_reg;
    always @(posedge clk) begin
        if (!rstn) begin
            pred_reg <= 4'd0;
        end else begin
            pred_reg <= next_pred;
        end
    end

    assign pred_out = pred_reg;

endmodule
