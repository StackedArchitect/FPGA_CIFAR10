`timescale 1ns / 1ps
//============================================================================
// 2D CNN Top Module for CIFAR-10 — Full-Precision + BatchNorm (Parallel)
//
// Architecture (v4 — 4 conv layers):
//   Conv2D_1 (3→32, 3×3, pad=1) → BN1 → ReLU → MaxPool2D(2×2)  → 16×16×32
//   Conv2D_2 (32→64, 3×3, pad=1) → BN2 → ReLU → MaxPool2D(2×2) → 8×8×64
//   Conv2D_3 (64→64, 3×3, pad=1) → BN3 → ReLU (no pool)        → 8×8×64
//   Conv2D_4 (64→64, 3×3, pad=1) → BN4 → ReLU (no pool)        → 8×8×64
//   Global Average Pool (8×8 → 1×1 per channel)                  → 64
//   FC1      (64→256) → BN5 → ReLU
//   FC2      (256→10) → logits (no BN)
//
// Layer chaining via rstn/done signals (sequential layer execution).
//
// Data flow (flat arrays, Q16.16 fixed-point):
//   data_in      [0:3071]         32×32×3       (channel-first)
//     ↓ conv1 + BN1 + pool1
//   pool1_out    [0:8191]         16×16×32
//     ↓ conv2 + BN2 + pool2
//   pool2_out    [0:4095]         8×8×64
//     ↓ conv3 + BN3 (no pool)
//   conv3_out    [0:4095]         8×8×64
//     ↓ conv4 + BN4 (no pool)
//   conv4_out    [0:4095]         8×8×64
//     ↓ GAP
//   gap_out      [0:63]           64
//     ↓ flatten + pad → fc1_in [0:103]
//   fc1_out      [0:255]          256
//     ↓ pad → fc2_in [0:295]
//   fc2_out      [0:9]            10 (logits)
//
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
module cnn2d_top_cifar #(
    // ---- Input ----
    parameter INPUT_H         = 32,
    parameter INPUT_W         = 32,
    parameter INPUT_CH        = 3,

    // ---- Conv1 ----
    parameter CONV1_OUT_CH    = 32,
    parameter CONV1_KERNEL    = 3,
    parameter CONV1_PAD       = 1,
    parameter CONV1_OUT_H     = INPUT_H + 2*CONV1_PAD - CONV1_KERNEL + 1,  // 32
    parameter CONV1_OUT_W     = INPUT_W + 2*CONV1_PAD - CONV1_KERNEL + 1,  // 32

    // ---- Pool1 ----
    parameter POOL1_SIZE      = 2,
    parameter POOL1_OUT_H     = CONV1_OUT_H / POOL1_SIZE,     // 16
    parameter POOL1_OUT_W     = CONV1_OUT_W / POOL1_SIZE,     // 16

    // ---- Conv2 ----
    parameter CONV2_IN_CH     = CONV1_OUT_CH,                 // 32
    parameter CONV2_OUT_CH    = 64,
    parameter CONV2_KERNEL    = 3,
    parameter CONV2_PAD       = 1,
    parameter CONV2_OUT_H     = POOL1_OUT_H + 2*CONV2_PAD - CONV2_KERNEL + 1,  // 16
    parameter CONV2_OUT_W     = POOL1_OUT_W + 2*CONV2_PAD - CONV2_KERNEL + 1,  // 16

    // ---- Pool2 ----
    parameter POOL2_SIZE      = 2,
    parameter POOL2_OUT_H     = CONV2_OUT_H / POOL2_SIZE,     // 8
    parameter POOL2_OUT_W     = CONV2_OUT_W / POOL2_SIZE,     // 8

    // ---- Conv3 (no pool) ----
    parameter CONV3_IN_CH     = CONV2_OUT_CH,                 // 64
    parameter CONV3_OUT_CH    = 64,
    parameter CONV3_KERNEL    = 3,
    parameter CONV3_PAD       = 1,
    parameter CONV3_OUT_H     = POOL2_OUT_H + 2*CONV3_PAD - CONV3_KERNEL + 1,  // 8
    parameter CONV3_OUT_W     = POOL2_OUT_W + 2*CONV3_PAD - CONV3_KERNEL + 1,  // 8

    // ---- Conv4 (no pool) ----
    parameter CONV4_IN_CH     = CONV3_OUT_CH,                 // 64
    parameter CONV4_OUT_CH    = 64,
    parameter CONV4_KERNEL    = 3,
    parameter CONV4_PAD       = 1,
    parameter CONV4_OUT_H     = CONV3_OUT_H + 2*CONV4_PAD - CONV4_KERNEL + 1,  // 8
    parameter CONV4_OUT_W     = CONV3_OUT_W + 2*CONV4_PAD - CONV4_KERNEL + 1,  // 8

    // ---- GAP ----
    parameter GAP_SHIFT       = 6,                            // log2(8×8) = 6
    parameter GAP_OUT         = CONV4_OUT_CH,                 // 64

    // ---- FC ----
    parameter FC1_IN          = GAP_OUT,                      // 64
    parameter FC1_OUT         = 256,
    parameter FC2_OUT         = 10,
    parameter PAD             = 20,

    // Padded widths for FC layers
    parameter FC1_WIDTH       = PAD + FC1_IN + PAD - 1,       // 103
    parameter FC2_WIDTH       = PAD + FC1_OUT + PAD - 1,      // 295

    // Parallelism
    parameter PARALLEL_CH     = 16,

    // Bit widths
    parameter BITS            = 31,    // Q16.16 input = 32-bit = [31:0]

    // Weight file paths for FC layers (BRAM ROM initialisation)
    parameter FC1_WEIGHT_FILE = "",
    parameter FC2_WEIGHT_FILE = ""
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Input image — 32×32×3 Q16.16 values (channel-first: CHW)
    input  wire signed [31:0]       data_in     [0 : INPUT_H * INPUT_W * INPUT_CH - 1],

    // Conv2D weights (flat, port arrays)
    input  wire signed [31:0]       conv1_w     [0 : CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL - 1],
    input  wire signed [31:0]       conv1_b     [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv2_w     [0 : CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL - 1],
    input  wire signed [31:0]       conv2_b     [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       conv3_w     [0 : CONV3_OUT_CH * CONV3_IN_CH * CONV3_KERNEL * CONV3_KERNEL - 1],
    input  wire signed [31:0]       conv3_b     [0 : CONV3_OUT_CH - 1],
    input  wire signed [31:0]       conv4_w     [0 : CONV4_OUT_CH * CONV4_IN_CH * CONV4_KERNEL * CONV4_KERNEL - 1],
    input  wire signed [31:0]       conv4_b     [0 : CONV4_OUT_CH - 1],

    // BatchNorm parameters (Q16.16 per output channel)
    input  wire signed [31:0]       conv1_bn_scale [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv1_bn_shift [0 : CONV1_OUT_CH - 1],
    input  wire signed [31:0]       conv2_bn_scale [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       conv2_bn_shift [0 : CONV2_OUT_CH - 1],
    input  wire signed [31:0]       conv3_bn_scale [0 : CONV3_OUT_CH - 1],
    input  wire signed [31:0]       conv3_bn_shift [0 : CONV3_OUT_CH - 1],
    input  wire signed [31:0]       conv4_bn_scale [0 : CONV4_OUT_CH - 1],
    input  wire signed [31:0]       conv4_bn_shift [0 : CONV4_OUT_CH - 1],

    // FC biases (weights stored internally by layer_seq_cifar)
    input  wire signed [31:0]       fc1_b       [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc2_b       [0 : FC2_OUT - 1],

    // FC1 BatchNorm parameters (BN5 in software)
    input  wire signed [31:0]       fc1_bn_scale [0 : FC1_OUT - 1],
    input  wire signed [31:0]       fc1_bn_shift [0 : FC1_OUT - 1],

    // Final output logits
    output wire signed [BITS+16:0]  cnn_out     [0 : FC2_OUT - 1]
);

    // ==================================================================
    //  Internal wires
    // ==================================================================

    // Pool1 output: 16×16×32 = 8192 values
    wire signed [BITS:0] pool1_out [0 : POOL1_OUT_H * POOL1_OUT_W * CONV1_OUT_CH - 1];
    wire                 pool1_done;

    // Pool2 output: 8×8×64 = 4096 values
    wire signed [BITS:0] pool2_out [0 : POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH - 1];
    wire                 pool2_done;

    // Conv3 output (no pool): 8×8×64 = 4096 values
    wire signed [BITS:0] conv3_out [0 : CONV3_OUT_H * CONV3_OUT_W * CONV3_OUT_CH - 1];
    wire                 conv3_done;

    // Conv4 output (no pool): 8×8×64 = 4096 values
    wire signed [BITS:0] conv4_out [0 : CONV4_OUT_H * CONV4_OUT_W * CONV4_OUT_CH - 1];
    wire                 conv4_done;

    // GAP output: 64 values
    wire signed [BITS:0] gap_out [0 : GAP_OUT - 1];
    wire                 gap_done;

    // FC1 input bus: PAD + FC1_IN + PAD = 104 values [0:103]
    wire signed [BITS:0] fc1_in [0 : FC1_WIDTH];

    // FC1 output
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;

    // FC2 input bus: PAD + FC1_OUT + PAD = 296 values [0:295]
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
    //  Conv1 + BN1 + Pool1: 32×32×3 → 32×32×32 → BN → ReLU → 16×16×32
    // ==================================================================
    conv_pool_2d_cifar #(
        .IN_H        (INPUT_H),
        .IN_W        (INPUT_W),
        .IN_CH       (INPUT_CH),
        .OUT_CH      (CONV1_OUT_CH),
        .KERNEL_H    (CONV1_KERNEL),
        .KERNEL_W    (CONV1_KERNEL),
        .PAD_SIZE    (CONV1_PAD),
        .POOL_H      (POOL1_SIZE),
        .POOL_W      (POOL1_SIZE),
        .HAS_POOL    (1),
        .PARALLEL_CH (PARALLEL_CH),
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
    //  Conv2 + BN2 + Pool2: 16×16×32 → 16×16×64 → BN → ReLU → 8×8×64
    // ==================================================================
    conv_pool_2d_cifar #(
        .IN_H        (POOL1_OUT_H),
        .IN_W        (POOL1_OUT_W),
        .IN_CH       (CONV2_IN_CH),
        .OUT_CH      (CONV2_OUT_CH),
        .KERNEL_H    (CONV2_KERNEL),
        .KERNEL_W    (CONV2_KERNEL),
        .PAD_SIZE    (CONV2_PAD),
        .POOL_H      (POOL2_SIZE),
        .POOL_W      (POOL2_SIZE),
        .HAS_POOL    (1),
        .PARALLEL_CH (PARALLEL_CH),
        .BITS        (BITS)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (pool1_done),    // Start when Conv1+Pool1 finishes
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
    //  Conv3 + BN3 (no pool): 8×8×64 → 8×8×64 → BN → ReLU → 8×8×64
    // ==================================================================
    conv_pool_2d_cifar #(
        .IN_H        (POOL2_OUT_H),
        .IN_W        (POOL2_OUT_W),
        .IN_CH       (CONV3_IN_CH),
        .OUT_CH      (CONV3_OUT_CH),
        .KERNEL_H    (CONV3_KERNEL),
        .KERNEL_W    (CONV3_KERNEL),
        .PAD_SIZE    (CONV3_PAD),
        .POOL_H      (1),               // Unused (HAS_POOL=0)
        .POOL_W      (1),               // Unused
        .HAS_POOL    (0),               // No pooling
        .PARALLEL_CH (PARALLEL_CH),
        .BITS        (BITS)
    ) u_conv_3 (
        .clk                (clk),
        .rstn               (pool2_done),    // Start when Conv2+Pool2 finishes
        .activation_function(1'b1),          // ReLU
        .data_in            (pool2_out),
        .weights            (conv3_w),
        .bias               (conv3_b),
        .bn_scale           (conv3_bn_scale),
        .bn_shift           (conv3_bn_shift),
        .data_out           (conv3_out),
        .done               (conv3_done)
    );


    // ==================================================================
    //  Conv4 + BN4 (no pool): 8×8×64 → 8×8×64 → BN → ReLU → 8×8×64
    // ==================================================================
    conv_pool_2d_cifar #(
        .IN_H        (CONV3_OUT_H),
        .IN_W        (CONV3_OUT_W),
        .IN_CH       (CONV4_IN_CH),
        .OUT_CH      (CONV4_OUT_CH),
        .KERNEL_H    (CONV4_KERNEL),
        .KERNEL_W    (CONV4_KERNEL),
        .PAD_SIZE    (CONV4_PAD),
        .POOL_H      (1),               // Unused (HAS_POOL=0)
        .POOL_W      (1),               // Unused
        .HAS_POOL    (0),               // No pooling
        .PARALLEL_CH (PARALLEL_CH),
        .BITS        (BITS)
    ) u_conv_4 (
        .clk                (clk),
        .rstn               (conv3_done),    // Start when Conv3 finishes
        .activation_function(1'b1),          // ReLU
        .data_in            (conv3_out),
        .weights            (conv4_w),
        .bias               (conv4_b),
        .bn_scale           (conv4_bn_scale),
        .bn_shift           (conv4_bn_shift),
        .data_out           (conv4_out),
        .done               (conv4_done)
    );


    // ==================================================================
    //  Global Average Pool: 8×8×64 → 64
    // ==================================================================
    global_avg_pool_cifar #(
        .IN_CH     (CONV4_OUT_CH),
        .IN_H      (CONV4_OUT_H),
        .IN_W      (CONV4_OUT_W),
        .GAP_SHIFT (GAP_SHIFT),
        .BITS      (BITS)
    ) u_gap (
        .clk      (clk),
        .rstn     (conv4_done),          // Start when Conv4 finishes
        .data_in  (conv4_out),
        .data_out (gap_out),
        .done     (gap_done)
    );


    // ==================================================================
    //  Flatten + Pad for FC1 input
    //  gap_out[0:63] → fc1_in[PAD : PAD+63] with zeros on both sides
    // ==================================================================
    genvar g;
    generate
        for (g = 0; g <= FC1_WIDTH; g = g + 1) begin : gen_fc1_pad
            if (g >= PAD && g < PAD + FC1_IN) begin : active
                assign fc1_in[g] = gap_out[g - PAD];
            end else begin : zero_pad
                assign fc1_in[g] = 32'sd0;
            end
        end
    endgenerate


    // ==================================================================
    //  FC1: 64 → 256, BN5 + ReLU
    //  Sequential MAC with internal BRAM weight ROM (layer_seq_cifar)
    // ==================================================================
    layer_seq_cifar #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .HAS_BN            (1),              // FC1 has BatchNorm (BN5)
        .WEIGHT_FILE       (FC1_WEIGHT_FILE)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (gap_done),      // Start when GAP finishes
        .activation_function(1'b1),          // ReLU
        .b                  (fc1_b),
        .data_in            (fc1_in),
        .bn_scale           (fc1_bn_scale),
        .bn_shift           (fc1_bn_shift),
        .data_out           (fc1_out_raw),
        .counter_donestatus (fc1_done)
    );


    // ==================================================================
    //  Pad FC1 output for FC2 input
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
    //  FC2: 256 → 10, no activation, no BN (raw logits)
    //  Sequential MAC with internal BRAM weight ROM (layer_seq_cifar)
    // ==================================================================
    localparam FC2_BITS = BITS + 8;  // FC1 output width

    layer_seq_cifar #(
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
