`timescale 1ns / 1ps
//============================================================================
// Synthesis Top — CIFAR-10 Full-Precision + BatchNorm (Parallel, v4)
//
// Wraps cnn2d_top_cifar with:
//   • ROM-initialised weight/bias/BN arrays (from .mem files)
//   • Argmax output (4-bit predicted class, 0–9)
//
// ROM sizing (Q16.16, 32-bit entries):
//   Conv1 weights:    864  → LUT-ROM or distributed RAM
//   Conv2 weights: 18,432  → BRAM (forced via ram_style = "block")
//   Conv3 weights: 36,864  → BRAM (forced)
//   Conv4 weights: 36,864  → BRAM (forced)
//   FC1 weights:   26,624  → internal BRAM (inside layer_seq_cifar)
//   FC2 weights:    2,960  → internal BRAM (inside layer_seq_cifar)
//   Input image:    3,072  → LUT-ROM
//   Biases + BN:    small per-channel arrays → LUT-ROM
//
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
module cnn2d_synth_top_cifar #(
    parameter BITS = 31
)(
    input  wire         clk,
    input  wire         rstn,
    output wire [3:0]   pred_out     // Predicted class (0–9)
);

    // ---- Architecture parameters ----
    localparam INPUT_H      = 32;
    localparam INPUT_W      = 32;
    localparam INPUT_CH     = 3;

    localparam CONV1_OUT_CH = 32;
    localparam CONV1_KERNEL = 3;
    localparam CONV1_PAD    = 1;

    localparam CONV2_IN_CH  = 32;
    localparam CONV2_OUT_CH = 64;
    localparam CONV2_KERNEL = 3;
    localparam CONV2_PAD    = 1;

    localparam CONV3_IN_CH  = 64;
    localparam CONV3_OUT_CH = 64;
    localparam CONV3_KERNEL = 3;
    localparam CONV3_PAD    = 1;

    localparam CONV4_IN_CH  = 64;
    localparam CONV4_OUT_CH = 64;
    localparam CONV4_KERNEL = 3;
    localparam CONV4_PAD    = 1;

    localparam FC1_IN       = 64;
    localparam FC1_OUT      = 256;
    localparam FC2_OUT      = 10;

    localparam PARALLEL_CH  = 16;

    // ---- Weight array sizes ----
    localparam CONV1_W_SIZE = CONV1_OUT_CH * INPUT_CH    * CONV1_KERNEL * CONV1_KERNEL;  // 864
    localparam CONV2_W_SIZE = CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL;  // 18,432
    localparam CONV3_W_SIZE = CONV3_OUT_CH * CONV3_IN_CH * CONV3_KERNEL * CONV3_KERNEL;  // 36,864
    localparam CONV4_W_SIZE = CONV4_OUT_CH * CONV4_IN_CH * CONV4_KERNEL * CONV4_KERNEL;  // 36,864

    // ================================================================
    //  Input image ROM — 3072 entries (3×32×32, channel-first)
    // ================================================================
    reg signed [31:0] data_in [0 : INPUT_H * INPUT_W * INPUT_CH - 1];
    initial $readmemh("data_in.mem", data_in);

    // ================================================================
    //  Conv1 weights + bias + BN — 864 weights + 32 bias + 32 BN
    // ================================================================
    reg signed [31:0] conv1_w [0 : CONV1_W_SIZE - 1];
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] conv1_bn_scale [0 : CONV1_OUT_CH - 1];
    reg signed [31:0] conv1_bn_shift [0 : CONV1_OUT_CH - 1];

    initial $readmemh("conv1_w.mem", conv1_w);
    initial $readmemh("conv1_b.mem", conv1_b);
    initial $readmemh("conv1_bn_scale.mem", conv1_bn_scale);
    initial $readmemh("conv1_bn_shift.mem", conv1_bn_shift);

    // ================================================================
    //  Conv2 weights + bias + BN — 18,432 weights → BRAM
    // ================================================================
    (* ram_style = "block" *) reg signed [31:0] conv2_w [0 : CONV2_W_SIZE - 1];
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] conv2_bn_scale [0 : CONV2_OUT_CH - 1];
    reg signed [31:0] conv2_bn_shift [0 : CONV2_OUT_CH - 1];

    initial $readmemh("conv2_w.mem", conv2_w);
    initial $readmemh("conv2_b.mem", conv2_b);
    initial $readmemh("conv2_bn_scale.mem", conv2_bn_scale);
    initial $readmemh("conv2_bn_shift.mem", conv2_bn_shift);

    // ================================================================
    //  Conv3 weights + bias + BN — 36,864 weights → BRAM
    // ================================================================
    (* ram_style = "block" *) reg signed [31:0] conv3_w [0 : CONV3_W_SIZE - 1];
    reg signed [31:0] conv3_b [0 : CONV3_OUT_CH - 1];
    reg signed [31:0] conv3_bn_scale [0 : CONV3_OUT_CH - 1];
    reg signed [31:0] conv3_bn_shift [0 : CONV3_OUT_CH - 1];

    initial $readmemh("conv3_w.mem", conv3_w);
    initial $readmemh("conv3_b.mem", conv3_b);
    initial $readmemh("conv3_bn_scale.mem", conv3_bn_scale);
    initial $readmemh("conv3_bn_shift.mem", conv3_bn_shift);

    // ================================================================
    //  Conv4 weights + bias + BN — 36,864 weights → BRAM
    // ================================================================
    (* ram_style = "block" *) reg signed [31:0] conv4_w [0 : CONV4_W_SIZE - 1];
    reg signed [31:0] conv4_b [0 : CONV4_OUT_CH - 1];
    reg signed [31:0] conv4_bn_scale [0 : CONV4_OUT_CH - 1];
    reg signed [31:0] conv4_bn_shift [0 : CONV4_OUT_CH - 1];

    initial $readmemh("conv4_w.mem", conv4_w);
    initial $readmemh("conv4_b.mem", conv4_b);
    initial $readmemh("conv4_bn_scale.mem", conv4_bn_scale);
    initial $readmemh("conv4_bn_shift.mem", conv4_bn_shift);

    // ================================================================
    //  FC biases + BN
    //  FC weights loaded internally by layer_seq_cifar
    // ================================================================
    reg signed [31:0] fc1_b [0 : FC1_OUT - 1];
    reg signed [31:0] fc2_b [0 : FC2_OUT - 1];
    reg signed [31:0] fc1_bn_scale [0 : FC1_OUT - 1];
    reg signed [31:0] fc1_bn_shift [0 : FC1_OUT - 1];

    initial $readmemh("fc1_b.mem", fc1_b);
    initial $readmemh("fc2_b.mem", fc2_b);
    initial $readmemh("fc1_bn_scale.mem", fc1_bn_scale);
    initial $readmemh("fc1_bn_shift.mem", fc1_bn_shift);

    // ================================================================
    //  DUT output
    // ================================================================
    wire signed [BITS+16:0] cnn_out [0 : FC2_OUT - 1];

    // ================================================================
    //  DUT instantiation
    // ================================================================
    cnn2d_top_cifar #(
        .INPUT_H      (INPUT_H),
        .INPUT_W      (INPUT_W),
        .INPUT_CH     (INPUT_CH),
        .CONV1_OUT_CH (CONV1_OUT_CH),
        .CONV1_KERNEL (CONV1_KERNEL),
        .CONV1_PAD    (CONV1_PAD),
        .CONV2_IN_CH  (CONV2_IN_CH),
        .CONV2_OUT_CH (CONV2_OUT_CH),
        .CONV2_KERNEL (CONV2_KERNEL),
        .CONV2_PAD    (CONV2_PAD),
        .CONV3_IN_CH  (CONV3_IN_CH),
        .CONV3_OUT_CH (CONV3_OUT_CH),
        .CONV3_KERNEL (CONV3_KERNEL),
        .CONV3_PAD    (CONV3_PAD),
        .CONV4_IN_CH  (CONV4_IN_CH),
        .CONV4_OUT_CH (CONV4_OUT_CH),
        .CONV4_KERNEL (CONV4_KERNEL),
        .CONV4_PAD    (CONV4_PAD),
        .FC1_OUT      (FC1_OUT),
        .FC2_OUT      (FC2_OUT),
        .PARALLEL_CH  (PARALLEL_CH),
        .BITS         (BITS),
        .FC1_WEIGHT_FILE("fc1_w.mem"),
        .FC2_WEIGHT_FILE("fc2_w.mem")
    ) dut (
        .clk             (clk),
        .rstn            (rstn),
        .data_in         (data_in),
        .conv1_w         (conv1_w),
        .conv1_b         (conv1_b),
        .conv2_w         (conv2_w),
        .conv2_b         (conv2_b),
        .conv3_w         (conv3_w),
        .conv3_b         (conv3_b),
        .conv4_w         (conv4_w),
        .conv4_b         (conv4_b),
        .conv1_bn_scale  (conv1_bn_scale),
        .conv1_bn_shift  (conv1_bn_shift),
        .conv2_bn_scale  (conv2_bn_scale),
        .conv2_bn_shift  (conv2_bn_shift),
        .conv3_bn_scale  (conv3_bn_scale),
        .conv3_bn_shift  (conv3_bn_shift),
        .conv4_bn_scale  (conv4_bn_scale),
        .conv4_bn_shift  (conv4_bn_shift),
        .fc1_b           (fc1_b),
        .fc2_b           (fc2_b),
        .fc1_bn_scale    (fc1_bn_scale),
        .fc1_bn_shift    (fc1_bn_shift),
        .cnn_out         (cnn_out)
    );

    // ================================================================
    //  Argmax — find predicted class (0–9)
    // ================================================================
    reg [3:0] argmax_idx;
    reg signed [BITS+16:0] argmax_val;
    integer k;

    always @(*) begin
        argmax_idx = 4'd0;
        argmax_val = cnn_out[0];
        for (k = 1; k < FC2_OUT; k = k + 1) begin
            if (cnn_out[k] > argmax_val) begin
                argmax_val = cnn_out[k];
                argmax_idx = k[3:0];
            end
        end
    end

    assign pred_out = argmax_idx;

endmodule
