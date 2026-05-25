`timescale 1ns / 1ps
//============================================================================
// 2D CNN Top Module for MNIST — Full-Precision + BatchNorm
//
// Architecture:
//   Conv2D_1 (1→4, 3×3, valid) → BN1 → ReLU → MaxPool2D(2×2)
//   Conv2D_2 (4→8, 3×3, valid) → BN2 → ReLU → MaxPool2D(2×2)
//   FC1      (200→32)          → BN3 → ReLU
//   FC2      (32→10)           → logits (no BN)
//
// Changes from baseline (no BN):
//   • New BN parameter ports: conv1_bn_scale, conv1_bn_shift,
//     conv2_bn_scale, conv2_bn_shift, fc1_bn_scale, fc1_bn_shift
//   • Uses conv_pool_2d_bn instead of conv_pool_2d
//   • Uses layer_seq_bn instead of layer_seq (with HAS_BN parameter)
//
// Data flow (flat arrays, Q16.16 fixed-point):
//   data_in    [0:783]            28×28×1       (row-major)
//     ↓ conv1 + BN1
//   conv1_out  [0:2703]           26×26×4       ([f][r][c])
//     ↓ pool1
//   pool1_out  [0:675]            13×13×4
//     ↓ conv2 + BN2
//   conv2_out  [0:967]            11×11×8
//     ↓ pool2
//   pool2_out  [0:199]            5×5×8
//     ↓ flatten (identity — already flat)
//   fc1_in     [0:239]            200 + 2×20 pad  (w/ padding for MAC)
//     ↓ fc1 + BN3
//   fc1_out    [0:31]             32
//     ↓ pad → fc2_in [0:71]       32 + 2×20 pad
//     ↓ fc2 (no BN)
//   fc2_out    [0:9]              10  (logits)
//============================================================================
module cnn2d_top_bn #(
    // ---- Input ----
    parameter INPUT_H         = 28,
    parameter INPUT_W         = 28,
    parameter INPUT_CH        = 1,

    // ---- Conv1 ----
    parameter CONV1_OUT_CH    = 4,
    parameter CONV1_KERNEL    = 3,
    parameter CONV1_OUT_H     = INPUT_H - CONV1_KERNEL + 1,   // 26
    parameter CONV1_OUT_W     = INPUT_W - CONV1_KERNEL + 1,   // 26

    // ---- Pool1 ----
    parameter POOL1_SIZE      = 2,
    parameter POOL1_OUT_H     = CONV1_OUT_H / POOL1_SIZE,     // 13
    parameter POOL1_OUT_W     = CONV1_OUT_W / POOL1_SIZE,     // 13

    // ---- Conv2 ----
    parameter CONV2_IN_CH     = CONV1_OUT_CH,                 // 4
    parameter CONV2_OUT_CH    = 8,
    parameter CONV2_KERNEL    = 3,
    parameter CONV2_OUT_H     = POOL1_OUT_H - CONV2_KERNEL + 1,  // 11
    parameter CONV2_OUT_W     = POOL1_OUT_W - CONV2_KERNEL + 1,  // 11

    // ---- Pool2 ----
    parameter POOL2_SIZE      = 2,
    parameter POOL2_OUT_H     = CONV2_OUT_H / POOL2_SIZE,     // 5
    parameter POOL2_OUT_W     = CONV2_OUT_W / POOL2_SIZE,     // 5

    // ---- FC ----
    parameter FLATTEN_SIZE    = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH,  // 200
    parameter FC1_OUT         = 32,
    parameter FC2_OUT         = 10,
    parameter PAD             = 20,

    // Padded widths for FC layers
    parameter FC1_WIDTH       = PAD + FLATTEN_SIZE + PAD - 1,  // 239
    parameter FC1_COUNTER_END = FC1_WIDTH - 3,                 // 236
    parameter FC2_WIDTH       = PAD + FC1_OUT + PAD - 1,       // 71
    parameter FC2_COUNTER_END = FC2_WIDTH - 3,                 // 68

    // Bit widths
    parameter BITS            = 31,     // Q16.16 input = 32-bit = [31:0]

    // Weight file paths for FC layers (BRAM ROM initialisation)
    parameter FC1_WEIGHT_FILE = "",
    parameter FC2_WEIGHT_FILE = ""
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input image — 28×28 Q16.16 values (row-major, 1 channel)
    input  wire signed [31:0]       data_in     [0 : INPUT_H * INPUT_W * INPUT_CH - 1],

    // Conv2D weights (flat)
    input  wire signed [31:0]       conv1_w     [0 : CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL - 1],
    input  wire signed [31:0]       conv1_b     [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv2_w     [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL - 1],
    input  wire signed [31:0]       conv2_b     [0 : CONV2_OUT_CH - 1],

    // BatchNorm parameters (Q16.16, per output channel)
    input  wire signed [31:0]       conv1_bn_scale [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv1_bn_shift [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv2_bn_scale [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       conv2_bn_shift [0 : CONV2_OUT_CH - 1],

    // FC biases (weights now stored as BRAM ROM inside layer_seq_bn)
    input  wire signed [31:0]       fc1_b       [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc2_b       [0 : FC2_OUT - 1],

    // FC1 BatchNorm parameters
    input  wire signed [31:0]       fc1_bn_scale [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc1_bn_shift [0 : FC1_OUT - 1],

    // Final output logits  (FC1 adds 8 bits → BITS+8; FC2 layer adds another 8 → BITS+16)
    output wire signed [BITS+16:0]  cnn_out     [0 : FC2_OUT - 1]
);

    // ==================================================================
    //  Internal wires
    // ==================================================================

    // Pool1 output: 13×13×4 = 676 values
    wire signed [BITS:0] pool1_out [0 : POOL1_OUT_H * POOL1_OUT_W * CONV1_OUT_CH - 1];
    wire                 pool1_done;

    // Pool2 output: 5×5×8 = 200 values
    wire signed [BITS:0] pool2_out [0 : POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH - 1];
    wire                 pool2_done;

    // FC1 input bus: PAD + FLATTEN_SIZE + PAD = 240 values [0:239]
    wire signed [BITS:0] fc1_in [0 : FC1_WIDTH];

    // FC1 output
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    // FC2 input bus: PAD + FC1_OUT + PAD = 72 values [0:71]
    wire signed [BITS+8:0] fc2_in [0 : FC2_WIDTH];

    // FC2 BN dummy wires (not used, HAS_BN=0)
    wire signed [31:0] fc2_bn_scale_dummy [0 : FC2_OUT - 1];
    wire signed [31:0] fc2_bn_shift_dummy [0 : FC2_OUT - 1];

    // Zero-tie the dummy BN wires
    genvar gi;
    generate
        for (gi = 0; gi < FC2_OUT; gi = gi + 1) begin : gen_fc2_bn_zero
            assign fc2_bn_scale_dummy[gi] = 32'sd0;
            assign fc2_bn_shift_dummy[gi] = 32'sd0;
        end
    endgenerate


    // ==================================================================
    //  Conv1 + BN1 + Pool1 (merged): 28×28×1 → 26×26×4 → BN → ReLU → 13×13×4
    // ==================================================================
    conv_pool_2d_bn #(
        .IN_H        (INPUT_H),
        .IN_W        (INPUT_W),
        .IN_CH       (INPUT_CH),
        .OUT_CH      (CONV1_OUT_CH),
        .KERNEL_H    (CONV1_KERNEL),
        .KERNEL_W    (CONV1_KERNEL),
        .POOL_H      (POOL1_SIZE),
        .POOL_W      (POOL1_SIZE),
        .BITS        (BITS)
    ) u_conv_pool_1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),          // ReLU
        .data_in            (data_in),
        .weights            (conv1_w),
        .bias               (conv1_b),
        .bn_scale           (conv1_bn_scale),
        .bn_shift           (conv1_bn_shift),
        .data_out           (pool1_out),
        .done               (pool1_done)
    );


    // ==================================================================
    //  Conv2 + BN2 + Pool2 (merged): 13×13×4 → 11×11×8 → BN → ReLU → 5×5×8
    // ==================================================================
    conv_pool_2d_bn #(
        .IN_H        (POOL1_OUT_H),
        .IN_W        (POOL1_OUT_W),
        .IN_CH       (CONV2_IN_CH),
        .OUT_CH      (CONV2_OUT_CH),
        .KERNEL_H    (CONV2_KERNEL),
        .KERNEL_W    (CONV2_KERNEL),
        .POOL_H      (POOL2_SIZE),
        .POOL_W      (POOL2_SIZE),
        .BITS        (BITS)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (pool1_done),    // Start when conv_pool_1 finishes
        .activation_function(1'b1),          // ReLU
        .data_in            (pool1_out),
        .weights            (conv2_w),
        .bias               (conv2_b),
        .bn_scale           (conv2_bn_scale),
        .bn_shift           (conv2_bn_shift),
        .data_out           (pool2_out),
        .done               (pool2_done)
    );


    // ==================================================================
    //  Flatten + Pad for FC1 input (unchanged from baseline)
    // ==================================================================
    genvar g;
    generate
        for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_pad
            if (g >= PAD && g < PAD + FLATTEN_SIZE) begin : active
                assign fc1_in[g] = pool2_out[g - PAD];
            end else begin : zero_pad
                assign fc1_in[g] = 32'sd0;
            end
        end
    endgenerate


    // ==================================================================
    //  FC1: 200 → 32, BN3 + ReLU
    //  Sequential MAC with internal BRAM weight ROM (layer_seq_bn)
    // ==================================================================
    layer_seq_bn #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .HAS_BN            (1),              // FC1 has BatchNorm
        .WEIGHT_FILE       (FC1_WEIGHT_FILE)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (pool2_done),    // Start when pool2 finishes
        .activation_function(1'b1),          // ReLU
        .b                  (fc1_b),
        .data_in            (fc1_in),
        .bn_scale           (fc1_bn_scale),
        .bn_shift           (fc1_bn_shift),
        .data_out           (fc1_out_raw),
        .counter_donestatus (fc1_done)
    );


    // ==================================================================
    //  Pad FC1 output for FC2 input (unchanged from baseline)
    // ==================================================================
    generate
        for (g = 0; g <= FC2_WIDTH; g = g + 1) begin : gen_fc2_pad
            if (g >= PAD && g < PAD + FC1_OUT) begin : active
                assign fc2_in[g] = fc1_out_raw[g - PAD];
            end else begin : zero_pad
                assign fc2_in[g] = {(BITS+9){1'b0}};
            end
        end
    endgenerate


    // ==================================================================
    //  FC2: 32 → 10, no activation, no BN (raw logits)
    //  Sequential MAC with internal BRAM weight ROM (layer_seq_bn)
    // ==================================================================
    localparam FC2_BITS = BITS + 8;  // FC1 output width

    layer_seq_bn #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_BITS        (FC2_BITS),
        .B_BITS            (31),
        .HAS_BN            (0),              // FC2 has NO BatchNorm
        .WEIGHT_FILE       (FC2_WEIGHT_FILE)
    ) u_fc2 (
        .clk                (clk),
        .rstn               (fc1_done),      // Start when FC1 finishes
        .activation_function(1'b0),          // No activation — raw logits
        .b                  (fc2_b),
        .data_in            (fc2_in),
        .bn_scale           (fc2_bn_scale_dummy),
        .bn_shift           (fc2_bn_shift_dummy),
        .data_out           (cnn_out),
        .counter_donestatus ()               // Not needed — last layer
    );

endmodule
