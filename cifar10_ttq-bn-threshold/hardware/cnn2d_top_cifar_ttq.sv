`timescale 1ns / 1ps
//============================================================================
// CNN 2D Top — CIFAR-10 TTQ + BatchNorm + DAAP + Hysteresis (Ping-Pong BRAM)
//
// Architecture (identical to baseline):
//   Conv1(3→32, pad=1) → BN1 → Pool1
//   Conv2(32→64, pad=1) → BN2 → Pool2
//   Conv3(64→64, pad=1) → BN3 (no pool)
//   Conv4(64→64, pad=1) → BN4 (no pool)
//   GAP(8×8→1×1) → FC1(64→256) → BN5 → FC2(256→10)
//
// Hysteresis Gating Integration:
//   • Instantiates three act_mask_gen_hyst mask generators:
//     - mask_gen_1 at Pool1 output (reads Buffer B, writes mask1_vector)
//     - mask_gen_2 at Pool2 output (reads Buffer A, writes mask2_vector)
//     - mask_gen_3 at Conv3 output (reads Buffer B, writes mask3_vector)
//   • Transitions through FSM phases:
//     - Phase 0: Conv1
//     - Phase 1: Mask Gen 1
//     - Phase 2: Conv2 (utilizes mask1_vector)
//     - Phase 3: Mask Gen 2
//     - Phase 4: Conv3 (utilizes mask2_vector)
//     - Phase 5: Mask Gen 3
//     - Phase 6: Conv4 (utilizes mask3_vector)
//     - Phase 7: GAP + FCs
//   • BRAM MUXes address/data lines during mask generator sub-phases.
//
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
(* KEEP_HIERARCHY = "yes" *) module cnn2d_top_cifar_ttq #(
    // ---- Input ----
    parameter INPUT_H         = 32,
    parameter INPUT_W         = 32,
    parameter INPUT_CH        = 3,

    // ---- Conv1 ----
    parameter CONV1_OUT_CH    = 32,
    parameter CONV1_KERNEL    = 3,
    parameter CONV1_PAD       = 1,
    parameter POOL1_SIZE      = 2,
    parameter POOL1_OUT_H     = (INPUT_H + 2*CONV1_PAD - CONV1_KERNEL + 1) / POOL1_SIZE,
    parameter POOL1_OUT_W     = (INPUT_W + 2*CONV1_PAD - CONV1_KERNEL + 1) / POOL1_SIZE,

    // ---- Conv2 ----
    parameter CONV2_IN_CH     = CONV1_OUT_CH,
    parameter CONV2_OUT_CH    = 64,
    parameter CONV2_KERNEL    = 3,
    parameter CONV2_PAD       = 1,
    parameter POOL2_SIZE      = 2,
    parameter POOL2_OUT_H     = (POOL1_OUT_H + 2*CONV2_PAD - CONV2_KERNEL + 1) / POOL2_SIZE,
    parameter POOL2_OUT_W     = (POOL1_OUT_W + 2*CONV2_PAD - CONV2_KERNEL + 1) / POOL2_SIZE,

    // ---- Conv3 ----
    parameter CONV3_IN_CH     = CONV2_OUT_CH,
    parameter CONV3_OUT_CH    = 64,
    parameter CONV3_KERNEL    = 3,
    parameter CONV3_PAD       = 1,
    parameter CONV3_OUT_H     = POOL2_OUT_H + 2*CONV3_PAD - CONV3_KERNEL + 1,
    parameter CONV3_OUT_W     = POOL2_OUT_W + 2*CONV3_PAD - CONV3_KERNEL + 1,

    // ---- Conv4 ----
    parameter CONV4_IN_CH     = CONV3_OUT_CH,
    parameter CONV4_OUT_CH    = 64,
    parameter CONV4_KERNEL    = 3,
    parameter CONV4_PAD       = 1,
    parameter CONV4_OUT_H     = CONV3_OUT_H + 2*CONV4_PAD - CONV4_KERNEL + 1,
    parameter CONV4_OUT_W     = CONV3_OUT_W + 2*CONV4_PAD - CONV4_KERNEL + 1,

    // ---- GAP ----
    parameter GAP_OUT         = CONV4_OUT_CH,
    parameter GAP_SHIFT       = 6,

    // ---- FC ----
    parameter FC1_IN          = GAP_OUT,
    parameter FC1_OUT         = 256,
    parameter FC2_OUT         = 10,
    parameter PAD             = 20,
    parameter FC1_WIDTH       = PAD + FC1_IN + PAD - 1,
    parameter FC2_WIDTH       = PAD + FC1_OUT + PAD - 1,

    // Parallelism — 4 filters per group
    parameter PARALLEL_CH     = 4,

    // Bit widths
    parameter BITS            = 31,

    // Weight file paths for FC layers (ternary codes)
    parameter FC1_WEIGHT_FILE = "",
    parameter FC2_WEIGHT_FILE = "",

    // Conv weight file paths (ternary codes)
    parameter CONV1_WEIGHT_FILE   = "",
    parameter CONV1_BIAS_FILE     = "",
    parameter CONV1_BN_SCALE_FILE = "",
    parameter CONV1_BN_SHIFT_FILE = "",
    parameter CONV1_WP_FILE       = "",
    parameter CONV1_WN_FILE       = "",

    parameter CONV2_WEIGHT_FILE   = "",
    parameter CONV2_BIAS_FILE     = "",
    parameter CONV2_BN_SCALE_FILE = "",
    parameter CONV2_BN_SHIFT_FILE = "",
    parameter CONV2_WP_FILE       = "",
    parameter CONV2_WN_FILE       = "",

    parameter CONV3_WEIGHT_FILE   = "",
    parameter CONV3_BIAS_FILE     = "",
    parameter CONV3_BN_SCALE_FILE = "",
    parameter CONV3_BN_SHIFT_FILE = "",
    parameter CONV3_WP_FILE       = "",
    parameter CONV3_WN_FILE       = "",

    parameter CONV4_WEIGHT_FILE   = "",
    parameter CONV4_BIAS_FILE     = "",
    parameter CONV4_BN_SCALE_FILE = "",
    parameter CONV4_BN_SHIFT_FILE = "",
    parameter CONV4_WP_FILE       = "",
    parameter CONV4_WN_FILE       = "",

    // FC bias/BN file paths
    parameter FC1_BIAS_FILE       = "",
    parameter FC2_BIAS_FILE       = "",
    parameter FC1_BN_SCALE_FILE   = "",
    parameter FC1_BN_SHIFT_FILE   = "",

    // FC TTQ scaling factors
    parameter FC1_WP_FILE         = "",
    parameter FC1_WN_FILE         = "",
    parameter FC2_WP_FILE         = "",
    parameter FC2_WN_FILE         = "",

    // DAAP activation threshold file paths
    parameter CONV1_ACT_THRESH_FILE = "",
    parameter CONV2_ACT_THRESH_FILE = "",
    parameter CONV3_ACT_THRESH_FILE = "",
    parameter CONV4_ACT_THRESH_FILE = "",
    parameter FC1_ACT_THRESH_FILE   = "",
    parameter FC2_ACT_THRESH_FILE   = "",



    // Input image file path
    parameter DATA_IN_FILE        = ""
)(
    input  wire                     clk,
    input  wire                     rstn,

    // Final output logits
    output wire signed [BITS+16:0]  cnn_out     [0 : FC2_OUT - 1]
);

    // ==================================================================
    //  Feature map sizes (max needed for ping-pong buffer sizing)
    // ==================================================================
    localparam DATA_IN_SIZE = INPUT_H * INPUT_W * INPUT_CH;        // 3072
    localparam POOL1_SIZE_T = POOL1_OUT_H * POOL1_OUT_W * CONV1_OUT_CH;  // 8192
    localparam POOL2_SIZE_T = POOL2_OUT_H * POOL2_OUT_W * CONV2_OUT_CH;  // 4096
    localparam CONV3_SIZE   = CONV3_OUT_H * CONV3_OUT_W * CONV3_OUT_CH;  // 4096
    localparam CONV4_SIZE   = CONV4_OUT_H * CONV4_OUT_W * CONV4_OUT_CH;  // 4096

    // Max feature map size across all layers = POOL1_SIZE_T = 8192
    localparam FMAP_BUF_SIZE = POOL1_SIZE_T;  // 8192 entries per buffer

    // ==================================================================
    //  Ping-Pong Feature Map BRAMs
    // ==================================================================
    (* ram_style = "block" *) reg signed [BITS:0] fmap_a [0 : FMAP_BUF_SIZE - 1];
    (* ram_style = "block" *) reg signed [BITS:0] fmap_b [0 : FMAP_BUF_SIZE - 1];

    // Load input image into buffer A
    initial $readmemh(DATA_IN_FILE, fmap_a);

    // ==================================================================
    //  Phase tracking (0..7)
    // ==================================================================
    reg [2:0] phase;

    wire pool1_done, pool2_done, conv3_done, conv4_done, gap_done;
    wire mask_gen1_done, mask_gen2_done, mask_gen3_done;

    always @(posedge clk) begin
        if (!rstn)
            phase <= 3'd0;
        else begin
            case (phase)
                3'd0: if (pool1_done) phase <= 3'd1;
                3'd1: if (mask_gen1_done) phase <= 3'd2;
                3'd2: if (pool2_done) phase <= 3'd3;
                3'd3: if (mask_gen2_done) phase <= 3'd4;
                3'd4: if (conv3_done) phase <= 3'd5;
                3'd5: if (mask_gen3_done) phase <= 3'd6;
                3'd6: if (conv4_done) phase <= 3'd7;
                default: ;
            endcase
        end
    end

    // Start triggers for mask generators
    reg mask_gen1_start, mask_gen2_start, mask_gen3_start;
    always @(posedge clk) begin
        if (!rstn) begin
            mask_gen1_start <= 1'b0;
            mask_gen2_start <= 1'b0;
            mask_gen3_start <= 1'b0;
        end else begin
            mask_gen1_start <= (phase == 3'd0 && pool1_done);
            mask_gen2_start <= (phase == 3'd2 && pool2_done);
            mask_gen3_start <= (phase == 3'd4 && conv3_done);
        end
    end

    // ==================================================================
    //  Conv module address/data wires
    // ==================================================================
    wire [31:0]          conv1_rd_addr, conv2_rd_addr, conv3_rd_addr, conv4_rd_addr;
    wire [31:0]          conv1_wr_addr, conv2_wr_addr, conv3_wr_addr, conv4_wr_addr;
    wire signed [BITS:0] conv1_wr_data, conv2_wr_data, conv3_wr_data, conv4_wr_data;
    wire                 conv1_wr_en,   conv2_wr_en,   conv3_wr_en,   conv4_wr_en;

    wire [31:0]          gap_rd_addr;

    // Mask generator read addresses
    wire [31:0]          mask_gen1_rd_addr, mask_gen2_rd_addr, mask_gen3_rd_addr;

    // Read data wires
    reg signed [BITS:0]  fmap_a_rd_data, fmap_b_rd_data;

    // ==================================================================
    //  BRAM A — Read address MUX
    // ==================================================================
    reg [31:0] bram_a_rd_addr;
    always @(*) begin
        case (phase)
            3'd0: bram_a_rd_addr = conv1_rd_addr;
            3'd3: bram_a_rd_addr = mask_gen2_rd_addr;  // Mask Gen 2 reads BRAM A (Pool2)
            3'd4: bram_a_rd_addr = conv3_rd_addr;
            3'd7: bram_a_rd_addr = gap_rd_addr;
            default: bram_a_rd_addr = 0;
        endcase
    end

    // BRAM A — Write MUX (even write phases write A)
    reg [31:0]          bram_a_wr_addr;
    reg signed [BITS:0] bram_a_wr_data;
    reg                 bram_a_wr_en;
    always @(*) begin
        case (phase)
            3'd2: begin // Phase 2: Conv2 writes to A (Pool2)
                bram_a_wr_addr = conv2_wr_addr;
                bram_a_wr_data = conv2_wr_data;
                bram_a_wr_en   = conv2_wr_en;
            end
            3'd6: begin // Phase 6: Conv4 writes to A (Conv4)
                bram_a_wr_addr = conv4_wr_addr;
                bram_a_wr_data = conv4_wr_data;
                bram_a_wr_en   = conv4_wr_en;
            end
            default: begin
                bram_a_wr_addr = 0;
                bram_a_wr_data = 0;
                bram_a_wr_en   = 0;
            end
        endcase
    end

    // BRAM A — Synchronous read + write
    always @(posedge clk) begin
        fmap_a_rd_data <= fmap_a[bram_a_rd_addr];
        if (bram_a_wr_en)
            fmap_a[bram_a_wr_addr] <= bram_a_wr_data;
    end

    // ==================================================================
    //  BRAM B — Read address MUX
    // ==================================================================
    reg [31:0] bram_b_rd_addr;
    always @(*) begin
        case (phase)
            3'd1:    bram_b_rd_addr = mask_gen1_rd_addr; // Mask Gen 1 reads BRAM B (Pool1)
            3'd2:    bram_b_rd_addr = conv2_rd_addr;
            3'd5:    bram_b_rd_addr = mask_gen3_rd_addr; // Mask Gen 3 reads BRAM B (Conv3)
            3'd6:    bram_b_rd_addr = conv4_rd_addr;
            default: bram_b_rd_addr = 0;
        endcase
    end

    // BRAM B — Write MUX (odd write phases write B)
    reg [31:0]          bram_b_wr_addr;
    reg signed [BITS:0] bram_b_wr_data;
    reg                 bram_b_wr_en;
    always @(*) begin
        case (phase)
            3'd0: begin // Phase 0: Conv1 writes to B (Pool1)
                bram_b_wr_addr = conv1_wr_addr;
                bram_b_wr_data = conv1_wr_data;
                bram_b_wr_en   = conv1_wr_en;
            end
            3'd4: begin // Phase 4: Conv3 writes to B (Conv3)
                bram_b_wr_addr = conv3_wr_addr;
                bram_b_wr_data = conv3_wr_data;
                bram_b_wr_en   = conv3_wr_en;
            end
            default: begin
                bram_b_wr_addr = 0;
                bram_b_wr_data = 0;
                bram_b_wr_en   = 0;
            end
        endcase
    end

    // BRAM B — Synchronous read + write
    always @(posedge clk) begin
        fmap_b_rd_data <= fmap_b[bram_b_rd_addr];
        if (bram_b_wr_en)
            fmap_b[bram_b_wr_addr] <= bram_b_wr_data;
    end

    // ==================================================================
    //  Read data routing
    // ==================================================================
    wire signed [BITS:0] conv1_rd_data = fmap_a_rd_data;
    wire signed [BITS:0] conv2_rd_data = fmap_b_rd_data;
    wire signed [BITS:0] conv3_rd_data = fmap_a_rd_data;
    wire signed [BITS:0] conv4_rd_data = fmap_b_rd_data;
    wire signed [BITS:0] gap_rd_data   = fmap_a_rd_data;

    // ==================================================================
    //  GAP output + FC wires
    // ==================================================================
    wire signed [BITS:0] gap_out [0 : GAP_OUT - 1];

    wire signed [BITS:0]   fc1_in [0 : FC1_WIDTH];
    wire signed [BITS+8:0] fc1_out_raw [0 : FC1_OUT - 1];
    wire                   fc1_done;
    wire signed [BITS+8:0] fc2_in [0 : FC2_WIDTH];

    // FC2 BN dummy wires (not used, HAS_BN=0)
    wire signed [31:0] fc2_bn_scale_dummy [0 : FC2_OUT - 1];
    wire signed [31:0] fc2_bn_shift_dummy [0 : FC2_OUT - 1];

    genvar gi;
    generate
        for (gi = 0; gi < FC2_OUT; gi = gi + 1) begin : gen_fc2_bn_zero
            assign fc2_bn_scale_dummy[gi] = 32'sd0;
            assign fc2_bn_shift_dummy[gi] = 32'sd0;
        end
    endgenerate

    // ==================================================================
    //  FC1 bias + BN ROMs (forced distributed)
    // ==================================================================
    (* ram_style = "distributed" *) reg signed [31:0] fc1_b_rom [0 : FC1_OUT - 1];
    (* ram_style = "distributed" *) reg signed [31:0] fc1_bns_rom [0 : FC1_OUT - 1];
    (* ram_style = "distributed" *) reg signed [31:0] fc1_bnsh_rom [0 : FC1_OUT - 1];
    initial $readmemh(FC1_BIAS_FILE,     fc1_b_rom);
    initial $readmemh(FC1_BN_SCALE_FILE, fc1_bns_rom);
    initial $readmemh(FC1_BN_SHIFT_FILE, fc1_bnsh_rom);

    // FC2 bias ROM (forced distributed)
    (* ram_style = "distributed" *) reg signed [31:0] fc2_b_rom [0 : FC2_OUT - 1];
    initial $readmemh(FC2_BIAS_FILE, fc2_b_rom);

    // ==================================================================
    //  FC Wp/Wn scalar ROMs
    // ==================================================================
    reg signed [31:0] fc1_wp_arr [0:0];
    reg signed [31:0] fc1_wn_arr [0:0];
    reg signed [31:0] fc2_wp_arr [0:0];
    reg signed [31:0] fc2_wn_arr [0:0];
    initial $readmemh(FC1_WP_FILE, fc1_wp_arr);
    initial $readmemh(FC1_WN_FILE, fc1_wn_arr);
    initial $readmemh(FC2_WP_FILE, fc2_wp_arr);
    initial $readmemh(FC2_WN_FILE, fc2_wn_arr);
    wire signed [31:0] fc1_wp = fc1_wp_arr[0];
    wire signed [31:0] fc1_wn = fc1_wn_arr[0];
    wire signed [31:0] fc2_wp = fc2_wp_arr[0];
    wire signed [31:0] fc2_wn = fc2_wn_arr[0];

    // ==================================================================
    //  Threshold Mask Outputs
    // ==================================================================
    wire [POOL1_SIZE_T-1:0] mask1_vector;
    wire [POOL2_SIZE_T-1:0] mask2_vector;
    wire [CONV3_SIZE-1:0]   mask3_vector;

    // ==================================================================
    //  Instantiate Mask Generators
    // ==================================================================
    act_mask_gen_thresh #(
        .IN_H       (16),
        .IN_W       (16),
        .IN_CH      (32),
        .BITS       (BITS),
        .LOG2_HW    (8),
        .THRESH_FILE(CONV1_ACT_THRESH_FILE)
    ) u_mask_gen_1 (
        .clk         (clk),
        .rstn        (rstn),
        .start       (mask_gen1_start),
        .bram_rd_addr(mask_gen1_rd_addr),
        .bram_rd_data(fmap_b_rd_data), // reads from BRAM B (Pool1 output)
        .mask_out    (mask1_vector),
        .done        (mask_gen1_done)
    );

    act_mask_gen_thresh #(
        .IN_H       (8),
        .IN_W       (8),
        .IN_CH      (64),
        .BITS       (BITS),
        .LOG2_HW    (6),
        .THRESH_FILE(CONV2_ACT_THRESH_FILE)
    ) u_mask_gen_2 (
        .clk         (clk),
        .rstn        (rstn),
        .start       (mask_gen2_start),
        .bram_rd_addr(mask_gen2_rd_addr),
        .bram_rd_data(fmap_a_rd_data), // reads from BRAM A (Pool2 output)
        .mask_out    (mask2_vector),
        .done        (mask_gen2_done)
    );

    act_mask_gen_thresh #(
        .IN_H       (8),
        .IN_W       (8),
        .IN_CH      (64),
        .BITS       (BITS),
        .LOG2_HW    (6),
        .THRESH_FILE(CONV3_ACT_THRESH_FILE)
    ) u_mask_gen_3 (
        .clk         (clk),
        .rstn        (rstn),
        .start       (mask_gen3_start),
        .bram_rd_addr(mask_gen3_rd_addr),
        .bram_rd_data(fmap_b_rd_data), // reads from BRAM B (Conv3 output)
        .mask_out    (mask3_vector),
        .done        (mask_gen3_done)
    );

    // ==================================================================
    //  Conv1 + BN1 + Pool1: 32×32×3 → 32×32×32 → BN → ReLU → 16×16×32
    //  (no input mask applied — pass constant 1s)
    // ==================================================================
    conv_pool_2d_cifar_ttq #(
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
        .BITS        (BITS),
        .MASK_SIZE   (DATA_IN_SIZE),
        .WEIGHT_FILE  (CONV1_WEIGHT_FILE),
        .BIAS_FILE    (CONV1_BIAS_FILE),
        .BN_SCALE_FILE(CONV1_BN_SCALE_FILE),
        .BN_SHIFT_FILE(CONV1_BN_SHIFT_FILE),
        .WP_FILE      (CONV1_WP_FILE),
        .WN_FILE      (CONV1_WN_FILE),
        .ACT_THRESH_FILE(CONV1_ACT_THRESH_FILE)
    ) u_conv_pool_1 (
        .clk                (clk),
        .rstn               (rstn),
        .activation_function(1'b1),
        .act_mask           ({(DATA_IN_SIZE){1'b1}}),
        .fm_rd_addr         (conv1_rd_addr),
        .fm_rd_data         (conv1_rd_data),
        .fm_wr_addr         (conv1_wr_addr),
        .fm_wr_data         (conv1_wr_data),
        .fm_wr_en           (conv1_wr_en),
        .done               (pool1_done)
    );


    // ==================================================================
    //  Conv2 + BN2 + Pool2: 16×16×32 → 16×16×64 → BN → ReLU → 8×8×64
    //  (uses mask1_vector)
    // ==================================================================
    conv_pool_2d_cifar_ttq #(
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
        .BITS        (BITS),
        .MASK_SIZE   (POOL1_SIZE_T),
        .WEIGHT_FILE  (CONV2_WEIGHT_FILE),
        .BIAS_FILE    (CONV2_BIAS_FILE),
        .BN_SCALE_FILE(CONV2_BN_SCALE_FILE),
        .BN_SHIFT_FILE(CONV2_BN_SHIFT_FILE),
        .WP_FILE      (CONV2_WP_FILE),
        .WN_FILE      (CONV2_WN_FILE),
        .ACT_THRESH_FILE(CONV2_ACT_THRESH_FILE)
    ) u_conv_pool_2 (
        .clk                (clk),
        .rstn               (phase >= 3'd2),
        .activation_function(1'b1),
        .act_mask           (mask1_vector),
        .fm_rd_addr         (conv2_rd_addr),
        .fm_rd_data         (conv2_rd_data),
        .fm_wr_addr         (conv2_wr_addr),
        .fm_wr_data         (conv2_wr_data),
        .fm_wr_en           (conv2_wr_en),
        .done               (pool2_done)
    );


    // ==================================================================
    //  Conv3 + BN3 (no pool): 8×8×64 → 8×8×64 → BN → ReLU → 8×8×64
    //  (uses mask2_vector)
    // ==================================================================
    conv_pool_2d_cifar_ttq #(
        .IN_H        (POOL2_OUT_H),
        .IN_W        (POOL2_OUT_W),
        .IN_CH       (CONV3_IN_CH),
        .OUT_CH      (CONV3_OUT_CH),
        .KERNEL_H    (CONV3_KERNEL),
        .KERNEL_W    (CONV3_KERNEL),
        .PAD_SIZE    (CONV3_PAD),
        .POOL_H      (1),
        .POOL_W      (1),
        .HAS_POOL    (0),
        .PARALLEL_CH (PARALLEL_CH),
        .BITS        (BITS),
        .MASK_SIZE   (POOL2_SIZE_T),
        .WEIGHT_FILE  (CONV3_WEIGHT_FILE),
        .BIAS_FILE    (CONV3_BIAS_FILE),
        .BN_SCALE_FILE(CONV3_BN_SCALE_FILE),
        .BN_SHIFT_FILE(CONV3_BN_SHIFT_FILE),
        .WP_FILE      (CONV3_WP_FILE),
        .WN_FILE      (CONV3_WN_FILE),
        .ACT_THRESH_FILE(CONV3_ACT_THRESH_FILE)
    ) u_conv_3 (
        .clk                (clk),
        .rstn               (phase >= 3'd4),
        .activation_function(1'b1),
        .act_mask           (mask2_vector),
        .fm_rd_addr         (conv3_rd_addr),
        .fm_rd_data         (conv3_rd_data),
        .fm_wr_addr         (conv3_wr_addr),
        .fm_wr_data         (conv3_wr_data),
        .fm_wr_en           (conv3_wr_en),
        .done               (conv3_done)
    );


    // ==================================================================
    //  Conv4 + BN4 (no pool): 8×8×64 → 8×8×64 → BN → ReLU → 8×8×64
    //  (uses mask3_vector)
    // ==================================================================
    conv_pool_2d_cifar_ttq #(
        .IN_H        (CONV3_OUT_H),
        .IN_W        (CONV3_OUT_W),
        .IN_CH       (CONV4_IN_CH),
        .OUT_CH      (CONV4_OUT_CH),
        .KERNEL_H    (CONV4_KERNEL),
        .KERNEL_W    (CONV4_KERNEL),
        .PAD_SIZE    (CONV4_PAD),
        .POOL_H      (1),
        .POOL_W      (1),
        .HAS_POOL    (0),
        .PARALLEL_CH (PARALLEL_CH),
        .BITS        (BITS),
        .MASK_SIZE   (CONV3_SIZE),
        .WEIGHT_FILE  (CONV4_WEIGHT_FILE),
        .BIAS_FILE    (CONV4_BIAS_FILE),
        .BN_SCALE_FILE(CONV4_BN_SCALE_FILE),
        .BN_SHIFT_FILE(CONV4_BN_SHIFT_FILE),
        .WP_FILE      (CONV4_WP_FILE),
        .WN_FILE      (CONV4_WN_FILE),
        .ACT_THRESH_FILE(CONV4_ACT_THRESH_FILE)
    ) u_conv_4 (
        .clk                (clk),
        .rstn               (phase >= 3'd6),
        .activation_function(1'b1),
        .act_mask           (mask3_vector),
        .fm_rd_addr         (conv4_rd_addr),
        .fm_rd_data         (conv4_rd_data),
        .fm_wr_addr         (conv4_wr_addr),
        .fm_wr_data         (conv4_wr_data),
        .fm_wr_en           (conv4_wr_en),
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
        .clk        (clk),
        .rstn       (phase >= 3'd7 && conv4_done),
        .fm_rd_addr (gap_rd_addr),
        .fm_rd_data (gap_rd_data),
        .data_out   (gap_out),
        .done       (gap_done)
    );


    // ==================================================================
    //  Flatten + Pad for FC1 input
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
    //  FC1: 64 → 256, BN5 + ReLU (TTQ weights + threshold)
    // ==================================================================
    layer_seq_cifar_ttq #(
        .NUM_NEURONS       (FC1_OUT),
        .LAYER_NEURON_WIDTH(FC1_WIDTH),
        .LAYER_BITS        (BITS),
        .B_BITS            (31),
        .HAS_BN            (1),
        .FORCE_BRAM        (0),
        .WEIGHT_FILE       (FC1_WEIGHT_FILE),
        .ACT_THRESH_FILE   (FC1_ACT_THRESH_FILE)
    ) u_fc1 (
        .clk                (clk),
        .rstn               (phase >= 3'd7 && gap_done),
        .activation_function(1'b1),
        .b                  (fc1_b_rom),
        .data_in            (fc1_in),
        .wp                 (fc1_wp),
        .wn                 (fc1_wn),
        .bn_scale           (fc1_bns_rom),
        .bn_shift           (fc1_bnsh_rom),
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
    //  FC2: 256 → 10, no activation, no BN (TTQ weights + threshold)
    // ==================================================================
    localparam FC2_BITS = BITS + 8;

    layer_seq_cifar_ttq #(
        .NUM_NEURONS       (FC2_OUT),
        .LAYER_NEURON_WIDTH(FC2_WIDTH),
        .LAYER_BITS        (FC2_BITS),
        .B_BITS            (31),
        .HAS_BN            (0),
        .FORCE_BRAM        (0),
        .WEIGHT_FILE       (FC2_WEIGHT_FILE),
        .ACT_THRESH_FILE   (FC2_ACT_THRESH_FILE)
    ) u_fc2 (
        .clk                (clk),
        .rstn               (phase >= 3'd7 && fc1_done),
        .activation_function(1'b0),
        .b                  (fc2_b_rom),
        .data_in            (fc2_in),
        .wp                 (fc2_wp),
        .wn                 (fc2_wn),
        .bn_scale           (fc2_bn_scale_dummy),
        .bn_shift           (fc2_bn_shift_dummy),
        .data_out           (cnn_out),
        .counter_donestatus ()
    );

endmodule
