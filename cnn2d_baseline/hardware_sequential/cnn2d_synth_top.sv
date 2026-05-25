`timescale 1ns / 1ps
//==============================================================================
// cnn2d_synth_top.sv  —  Synthesizable top wrapper for cnn2d_top (2D CNN)
//
// Architecture : Conv2d(3×3, 4 filters) → ReLU → MaxPool2d(2×2) →
//                Conv2d(3×3, 8 filters) → ReLU → MaxPool2d(2×2) →
//                FC(32) → FC(10)    [Q16.16 fixed-point, 28×28 input]
// Compute DUT  : cnn2d_top.sv  (unchanged)
// Python model : 98.35 % accuracy on MNIST test set
//
// Synthesis notes
// ---------------
//  • Weight arrays are internal ROMs initialized from .mem files.
//    • conv1_w (36 words)   — LUT-ROM
//    • conv2_w (288 words)  — LUT-ROM
//    • fc1_w   (7680 words) — BRAM (Vivado auto-infers)
//    • fc2_w   (720 words)  — LUT-ROM or BRAM
//  • pixel_in is a real external port — Vivado analyzes actual timing paths.
//  • 2D arrays fc1_w / fc2_w are loaded row-by-row by $readmemh (outer
//    index first), matching the testbench convention.
//  • No logic in cnn2d_top.sv or any sub-module is changed.
//  • Adjust $readmemh paths if your Vivado project root differs from the
//    repository root.
//==============================================================================

module cnn2d_synth_top_seq (
    input  wire        clk,
    input  wire        rstn,
    // Inference result: argmax of 10 output logits (class 0-9)
    output reg  [3:0]  pred_out
);
    // Input image ROM — loaded from .mem; eliminates the impossible 25K-bit I/O bus
    // so Vivado can route the design and analyse real internal timing paths.
    reg signed [31:0] pixel_in [0 : 28*28 - 1];

    // -------------------------------------------------------------------------
    // Parameters (matching cnn2d_top.sv defaults)
    // -------------------------------------------------------------------------
    localparam INPUT_H       = 28;
    localparam INPUT_W       = 28;
    localparam INPUT_CH      = 1;

    localparam CONV1_OUT_CH  = 4;
    localparam CONV1_KERNEL  = 3;

    localparam POOL1_SIZE    = 2;

    localparam CONV2_IN_CH   = CONV1_OUT_CH;   // 4
    localparam CONV2_OUT_CH  = 8;
    localparam CONV2_KERNEL  = 3;

    localparam POOL2_SIZE    = 2;

    localparam FC1_OUT       = 32;
    localparam FC2_OUT       = 10;
    localparam PAD           = 20;
    localparam BITS          = 31;

    // Derived (same as cnn2d_top.sv)
    localparam CONV1_OUT_H   = INPUT_H - CONV1_KERNEL + 1;         // 26
    localparam CONV1_OUT_W   = INPUT_W - CONV1_KERNEL + 1;         // 26
    localparam POOL1_OUT_H   = CONV1_OUT_H / POOL1_SIZE;           // 13
    localparam POOL1_OUT_W   = CONV1_OUT_W / POOL1_SIZE;           // 13
    localparam CONV2_OUT_H   = POOL1_OUT_H - CONV2_KERNEL + 1;     // 11
    localparam CONV2_OUT_W   = POOL1_OUT_W - CONV2_KERNEL + 1;     // 11
    localparam POOL2_OUT_H   = CONV2_OUT_H / POOL2_SIZE;           // 5
    localparam POOL2_OUT_W   = CONV2_OUT_W / POOL2_SIZE;           // 5
    localparam FLATTEN_SIZE  = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH; // 200
    localparam FC1_WIDTH     = PAD + FLATTEN_SIZE + PAD - 1;       // 239
    localparam FC2_WIDTH     = PAD + FC1_OUT + PAD - 1;            // 71

    localparam CONV1_W_SIZE  = CONV1_OUT_CH * INPUT_CH * CONV1_KERNEL * CONV1_KERNEL;  // 36
    localparam CONV2_W_SIZE  = CONV2_OUT_CH * CONV2_IN_CH * CONV2_KERNEL * CONV2_KERNEL; // 288

    localparam OUT_BITS      = BITS + 16; // 47  (FC1 → BITS+8, FC2 layer → BITS+16)

    // -------------------------------------------------------------------------
    // Weight ROMs initialized from .mem files
    // Conv weights: small — stay as LUT-ROM (36 + 288 entries)
    // FC weights:   large — now stored as BRAM inside layer_seq modules
    // Adjust paths relative to your Vivado project root (.xpr location)
    // -------------------------------------------------------------------------
    reg signed [31:0] conv1_w [0 : CONV1_W_SIZE - 1];            //  36 entries
    reg signed [31:0] conv1_b [0 : CONV1_OUT_CH - 1];            //   4 entries
    reg signed [31:0] conv2_w [0 : CONV2_W_SIZE - 1];            // 288 entries
    reg signed [31:0] conv2_b [0 : CONV2_OUT_CH - 1];            //   8 entries
    reg signed [31:0] fc1_b   [0 : FC1_OUT - 1];                 //  32 entries
    reg signed [31:0] fc2_b   [0 : FC2_OUT - 1];                 //  10 entries

    initial begin
        $readmemh("cnn2d_weights/data_in.mem",  pixel_in);
        $readmemh("cnn2d_weights/conv1_w.mem", conv1_w);
        $readmemh("cnn2d_weights/conv1_b.mem", conv1_b);
        $readmemh("cnn2d_weights/conv2_w.mem", conv2_w);
        $readmemh("cnn2d_weights/conv2_b.mem", conv2_b);
        $readmemh("cnn2d_weights/fc1_b.mem",   fc1_b);
        $readmemh("cnn2d_weights/fc2_b.mem",   fc2_b);
    end

    // -------------------------------------------------------------------------
    // DUT instantiation — FC weights loaded internally via BRAM ROM
    // -------------------------------------------------------------------------
    wire signed [OUT_BITS:0] cnn_out [0 : FC2_OUT - 1];

    cnn2d_top_seq #(
        .INPUT_H        (INPUT_H),
        .INPUT_W        (INPUT_W),
        .INPUT_CH       (INPUT_CH),
        .CONV1_OUT_CH   (CONV1_OUT_CH),
        .CONV1_KERNEL   (CONV1_KERNEL),
        .POOL1_SIZE     (POOL1_SIZE),
        .CONV2_OUT_CH   (CONV2_OUT_CH),
        .CONV2_KERNEL   (CONV2_KERNEL),
        .POOL2_SIZE     (POOL2_SIZE),
        .FC1_OUT        (FC1_OUT),
        .FC2_OUT        (FC2_OUT),
        .PAD            (PAD),
        .BITS           (BITS),
        .FC1_WEIGHT_FILE("cnn2d_weights/fc1_w.mem"),
        .FC2_WEIGHT_FILE("cnn2d_weights/fc2_w.mem")
    ) u_cnn2d (
        .clk     (clk),
        .rstn    (rstn),
        .data_in (pixel_in),
        .conv1_w (conv1_w),
        .conv1_b (conv1_b),
        .conv2_w (conv2_w),
        .conv2_b (conv2_b),
        .fc1_b   (fc1_b),
        .fc2_b   (fc2_b),
        .cnn_out (cnn_out)
    );

    // -------------------------------------------------------------------------
    // Registered argmax (combinational comparison, registered on posedge clk)
    // -------------------------------------------------------------------------
    reg  signed [OUT_BITS:0] best_val;
    reg  [3:0]               best_idx;
    integer ai;
    always @(*) begin
        best_idx = 4'd0;
        best_val = cnn_out[0];
        for (ai = 1; ai < 10; ai = ai + 1) begin
            if ($signed(cnn_out[ai]) > best_val) begin
                best_idx = ai[3:0];
                best_val = cnn_out[ai];
            end
        end
    end

    always @(posedge clk or negedge rstn) begin
        if (!rstn) pred_out <= 4'd0;
        else       pred_out <= best_idx;
    end

endmodule
