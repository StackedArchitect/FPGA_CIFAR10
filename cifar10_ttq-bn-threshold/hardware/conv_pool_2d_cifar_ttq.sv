`timescale 1ns / 1ps
//============================================================================
// Conv2D + BatchNorm + Optional MaxPool — CIFAR-10 TTQ + DAAP + Hysteresis Gating
//
// Combines:
//   • CIFAR-10 baseline's BRAM ping-pong interface (fm_rd_addr/data,
//     fm_wr_addr/data/en) and PARALLEL_CH=4 parallel filter processing
//     with group iteration.
//   • MNIST TTQ's split accumulator pattern (pos_acc/neg_acc) and
//     S_CONV_SCALE state (Wp/Wn multiply + bias).
//   • DAAP activation threshold: per-filter Q16.16 threshold ROM.
//   • Spatial Hysteresis mask: 1-cycle-delayed act_mask check.
//     When mask_d == 0 or |input| < group_min_thresh, the tap is skipped.
//
// Weight ROM: 2-bit ternary codes (2'b01=+1, 2'b11=-1, 2'b00=0).
//   No DSP multipliers in the convolution compute path.
//   DSPs used only in S_CONV_SCALE (Wp*pos_acc, Wn*neg_acc) and
//   S_CONV_BN (bn_scale * biased_q16).
//
// Fixed-point: Q16.16
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
(* KEEP_HIERARCHY = "yes" *) module conv_pool_2d_cifar_ttq #(
    parameter IN_H        = 32,
    parameter IN_W        = 32,
    parameter IN_CH       = 3,
    parameter OUT_CH      = 32,
    parameter KERNEL_H    = 3,
    parameter KERNEL_W    = 3,
    parameter PAD_SIZE    = 1,
    parameter POOL_H      = 2,
    parameter POOL_W      = 2,
    parameter HAS_POOL    = 1,
    parameter PARALLEL_CH = 4,
    parameter CONV_OUT_H  = IN_H + 2*PAD_SIZE - KERNEL_H + 1,
    parameter CONV_OUT_W  = IN_W + 2*PAD_SIZE - KERNEL_W + 1,
    parameter POOL_OUT_H  = CONV_OUT_H / POOL_H,
    parameter POOL_OUT_W  = CONV_OUT_W / POOL_W,
    parameter BITS        = 31,
    parameter MASK_SIZE   = IN_CH * IN_H * IN_W,

    // Weight/BN file paths — loaded via $readmemh into internal ROMs.
    parameter WEIGHT_FILE     = "",   // 2-bit ternary codes
    parameter BIAS_FILE       = "",
    parameter BN_SCALE_FILE   = "",
    parameter BN_SHIFT_FILE   = "",
    parameter WP_FILE         = "",   // Single-entry Q16.16 positive scale
    parameter WN_FILE         = "",   // Single-entry Q16.16 negative scale
    parameter ACT_THRESH_FILE = ""    // Per-filter Q16.16 threshold (OUT_CH entries)
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     activation_function,

    // Hysteresis activation mask input
    input  wire [MASK_SIZE-1:0]     act_mask,

    // BRAM read port — reads from previous layer's feature map
    output wire [31:0]              fm_rd_addr,
    input  wire signed [BITS:0]     fm_rd_data,

    // BRAM write port — writes to this layer's feature map
    output reg  [31:0]              fm_wr_addr,
    output reg  signed [BITS:0]     fm_wr_data,
    output reg                      fm_wr_en,

    output reg                      done
);

    // ================================================================
    //  Internal Bias / BN ROMs (file-loaded, distributed)
    // ================================================================
    localparam TOTAL_W = OUT_CH * IN_CH * KERNEL_H * KERNEL_W;

    // Bias ROM (forced distributed — max 64 entries, saves BRAM)
    (* ram_style = "distributed" *) reg signed [31:0] b_rom [0 : OUT_CH - 1];
    initial $readmemh(BIAS_FILE, b_rom);

    // BN scale ROM (forced distributed)
    (* ram_style = "distributed" *) reg signed [31:0] bns_rom [0 : OUT_CH - 1];
    initial $readmemh(BN_SCALE_FILE, bns_rom);

    // BN shift ROM (forced distributed)
    (* ram_style = "distributed" *) reg signed [31:0] bnsh_rom [0 : OUT_CH - 1];
    initial $readmemh(BN_SHIFT_FILE, bnsh_rom);

    // ================================================================
    //  Wp / Wn scaling factors (single-entry Q16.16 files)
    // ================================================================
    reg signed [31:0] wp_arr [0:0];
    reg signed [31:0] wn_arr [0:0];
    initial $readmemh(WP_FILE, wp_arr);
    initial $readmemh(WN_FILE, wn_arr);
    wire signed [31:0] wp_val = wp_arr[0];
    wire signed [31:0] wn_val = wn_arr[0];

    // ================================================================
    //  Per-filter activation threshold — always present
    //  When all zeros: DAAP skip is never triggered → pure TTQ behavior
    // ================================================================
    (* ram_style = "distributed" *)
    reg signed [31:0] act_thresh_rom [0 : OUT_CH - 1];
    initial begin
        if (ACT_THRESH_FILE != "") begin
            $readmemh(ACT_THRESH_FILE, act_thresh_rom);
        end else begin
            for (integer k = 0; k < OUT_CH; k = k + 1) begin
                act_thresh_rom[k] = 32'sd0;
            end
        end
    end

    // ================================================================
    //  Constants
    // ================================================================
    localparam TAP_COUNT      = IN_CH * KERNEL_H * KERNEL_W;
    localparam SUB_W_SIZE     = TOTAL_W / PARALLEL_CH;

    localparam CONV_POSITIONS = CONV_OUT_H * CONV_OUT_W;
    localparam POOL_OUT_POS   = POOL_OUT_H * POOL_OUT_W;
    localparam POOL_ELEMENTS  = POOL_H * POOL_W;
    localparam NUM_GROUPS     = OUT_CH / PARALLEL_CH;
    localparam OUT_POSITIONS  = HAS_POOL ? POOL_OUT_POS : CONV_POSITIONS;

    // ================================================================
    //  Conv buffer — PARALLEL_CH filters (reused per group)
    //  Only used when HAS_POOL=1 for intermediate conv results.
    // ================================================================
    (* ram_style = "block" *) reg signed [BITS:0] conv_buf [0 : PARALLEL_CH * CONV_POSITIONS - 1];

    // ================================================================
    //  State machine — 9 states (4-bit register)
    // ================================================================
    localparam S_IDLE         = 4'd0;
    localparam S_CONV_COMPUTE = 4'd1;
    localparam S_CONV_DRAIN   = 4'd2;
    localparam S_CONV_SCALE   = 4'd3;   // Wp/Wn multiply + bias add
    localparam S_CONV_BN      = 4'd4;
    localparam S_CONV_STORE   = 4'd5;
    localparam S_POOL_COMPARE = 4'd6;
    localparam S_POOL_STORE   = 4'd7;
    localparam S_DONE         = 4'd8;

    reg [3:0] state;

    // ================================================================
    //  Group counter
    // ================================================================
    reg [31:0] group_idx;
    wire [31:0] group_base;
    assign group_base = group_idx * PARALLEL_CH;

    // ================================================================
    //  Minimum threshold across PARALLEL_CH filters in current group
    // ================================================================
    wire signed [31:0] thresh_0 = act_thresh_rom[group_base + 0];
    wire signed [31:0] thresh_1 = act_thresh_rom[group_base + 1];
    wire signed [31:0] thresh_2 = act_thresh_rom[group_base + 2];
    wire signed [31:0] thresh_3 = act_thresh_rom[group_base + 3];
    wire signed [31:0] min_01   = (thresh_0 < thresh_1) ? thresh_0 : thresh_1;
    wire signed [31:0] min_23   = (thresh_2 < thresh_3) ? thresh_2 : thresh_3;
    wire signed [31:0] group_min_thresh = (min_01 < min_23) ? min_01 : min_23;

    // ================================================================
    //  Conv counters
    // ================================================================
    reg [31:0] conv_out_row;
    reg [31:0] conv_out_col;
    reg [31:0] conv_pos;
    reg [31:0] ch_cnt;
    reg [31:0] kr_cnt;
    reg [31:0] kc_cnt;

    // BN/Store loop counter (0..PARALLEL_CH-1)
    reg [31:0] store_cnt;

    // ================================================================
    //  Pool counters
    // ================================================================
    reg [31:0] filter_idx;
    reg [31:0] pool_out_row;
    reg [31:0] pool_out_col;
    reg [31:0] pool_pos;
    reg [31:0] pool_r_cnt;
    reg [31:0] pool_c_cnt;
    reg [31:0] pool_counter;

    // ================================================================
    //  Combinational Lookahead Skip Logic
    // ================================================================
    
    // Tap 0 (Current)
    wire [31:0] ch_0 = ch_cnt;
    wire [31:0] kr_0 = kr_cnt;
    wire [31:0] kc_0 = kc_cnt;
    wire        eoc_0 = (kc_0 == KERNEL_W - 1) && (kr_0 == KERNEL_H - 1) && (ch_0 == IN_CH - 1);
    
    // Tap 1
    wire [31:0] kc_1 = (kc_0 == KERNEL_W - 1) ? 32'd0 : kc_0 + 32'd1;
    wire [31:0] kr_1 = (kc_0 == KERNEL_W - 1) ? ((kr_0 == KERNEL_H - 1) ? 32'd0 : kr_0 + 32'd1) : kr_0;
    wire [31:0] ch_1 = (kc_0 == KERNEL_W - 1 && kr_0 == KERNEL_H - 1) ? ch_0 + 32'd1 : ch_0;
    wire        eoc_1 = (kc_1 == KERNEL_W - 1) && (kr_1 == KERNEL_H - 1) && (ch_1 == IN_CH - 1);

    // Tap 2
    wire [31:0] kc_2 = (kc_1 == KERNEL_W - 1) ? 32'd0 : kc_1 + 32'd1;
    wire [31:0] kr_2 = (kc_1 == KERNEL_W - 1) ? ((kr_1 == KERNEL_H - 1) ? 32'd0 : kr_1 + 32'd1) : kr_1;
    wire [31:0] ch_2 = (kc_1 == KERNEL_W - 1 && kr_1 == KERNEL_H - 1) ? ch_1 + 32'd1 : ch_1;
    wire        eoc_2 = (kc_2 == KERNEL_W - 1) && (kr_2 == KERNEL_H - 1) && (ch_2 == IN_CH - 1);

    // Tap 3
    wire [31:0] kc_3 = (kc_2 == KERNEL_W - 1) ? 32'd0 : kc_2 + 32'd1;
    wire [31:0] kr_3 = (kc_2 == KERNEL_W - 1) ? ((kr_2 == KERNEL_H - 1) ? 32'd0 : kr_2 + 32'd1) : kr_2;
    wire [31:0] ch_3 = (kc_2 == KERNEL_W - 1 && kr_2 == KERNEL_H - 1) ? ch_2 + 32'd1 : ch_2;
    wire        eoc_3 = (kc_3 == KERNEL_W - 1) && (kr_3 == KERNEL_H - 1) && (ch_3 == IN_CH - 1);

    // Tap 4
    wire [31:0] kc_4 = (kc_3 == KERNEL_W - 1) ? 32'd0 : kc_3 + 32'd1;
    wire [31:0] kr_4 = (kc_3 == KERNEL_W - 1) ? ((kr_3 == KERNEL_H - 1) ? 32'd0 : kr_3 + 32'd1) : kr_3;
    wire [31:0] ch_4 = (kc_3 == KERNEL_W - 1 && kr_3 == KERNEL_H - 1) ? ch_3 + 32'd1 : ch_3;
    wire        eoc_4 = (kc_4 == KERNEL_W - 1) && (kr_4 == KERNEL_H - 1) && (ch_4 == IN_CH - 1);

    // Flat ROM weights address for each tap index
    wire [31:0] tap_idx_0 = ch_0 * (KERNEL_H * KERNEL_W) + kr_0 * KERNEL_W + kc_0;
    wire [31:0] tap_idx_1 = ch_1 * (KERNEL_H * KERNEL_W) + kr_1 * KERNEL_W + kc_1;
    wire [31:0] tap_idx_2 = ch_2 * (KERNEL_H * KERNEL_W) + kr_2 * KERNEL_W + kc_2;
    wire [31:0] tap_idx_3 = ch_3 * (KERNEL_H * KERNEL_W) + kr_3 * KERNEL_W + kc_3;

    wire [31:0] w_addr_0 = group_idx * TAP_COUNT + tap_idx_0;
    wire [31:0] w_addr_1 = group_idx * TAP_COUNT + tap_idx_1;
    wire [31:0] w_addr_2 = group_idx * TAP_COUNT + tap_idx_2;
    wire [31:0] w_addr_3 = group_idx * TAP_COUNT + tap_idx_3;

    wire [31:0] tap_idx = tap_idx_0;

    // Address & Bounds calculation
    wire [31:0] pad_row_sum_0 = conv_out_row + kr_0;
    wire [31:0] pad_col_sum_0 = conv_out_col + kc_0;
    wire in_row_valid_0 = (pad_row_sum_0 >= PAD_SIZE) && (pad_row_sum_0 < PAD_SIZE + IN_H);
    wire in_col_valid_0 = (pad_col_sum_0 >= PAD_SIZE) && (pad_col_sum_0 < PAD_SIZE + IN_W);
    wire bounds_0 = in_row_valid_0 && in_col_valid_0;
    wire [31:0] addr_0 = ch_0 * (IN_H * IN_W) + (pad_row_sum_0 - PAD_SIZE) * IN_W + (pad_col_sum_0 - PAD_SIZE);

    wire [31:0] pad_row_sum_1 = conv_out_row + kr_1;
    wire [31:0] pad_col_sum_1 = conv_out_col + kc_1;
    wire in_row_valid_1 = (pad_row_sum_1 >= PAD_SIZE) && (pad_row_sum_1 < PAD_SIZE + IN_H);
    wire in_col_valid_1 = (pad_col_sum_1 >= PAD_SIZE) && (pad_col_sum_1 < PAD_SIZE + IN_W);
    wire bounds_1 = in_row_valid_1 && in_col_valid_1;
    wire [31:0] addr_1 = ch_1 * (IN_H * IN_W) + (pad_row_sum_1 - PAD_SIZE) * IN_W + (pad_col_sum_1 - PAD_SIZE);

    wire [31:0] pad_row_sum_2 = conv_out_row + kr_2;
    wire [31:0] pad_col_sum_2 = conv_out_col + kc_2;
    wire in_row_valid_2 = (pad_row_sum_2 >= PAD_SIZE) && (pad_row_sum_2 < PAD_SIZE + IN_H);
    wire in_col_valid_2 = (pad_col_sum_2 >= PAD_SIZE) && (pad_col_sum_2 < PAD_SIZE + IN_W);
    wire bounds_2 = in_row_valid_2 && in_col_valid_2;
    wire [31:0] addr_2 = ch_2 * (IN_H * IN_W) + (pad_row_sum_2 - PAD_SIZE) * IN_W + (pad_col_sum_2 - PAD_SIZE);

    wire [31:0] pad_row_sum_3 = conv_out_row + kr_3;
    wire [31:0] pad_col_sum_3 = conv_out_col + kc_3;
    wire in_row_valid_3 = (pad_row_sum_3 >= PAD_SIZE) && (pad_row_sum_3 < PAD_SIZE + IN_H);
    wire in_col_valid_3 = (pad_col_sum_3 >= PAD_SIZE) && (pad_col_sum_3 < PAD_SIZE + IN_W);
    wire bounds_3 = in_row_valid_3 && in_col_valid_3;
    wire [31:0] addr_3 = ch_3 * (IN_H * IN_W) + (pad_row_sum_3 - PAD_SIZE) * IN_W + (pad_col_sum_3 - PAD_SIZE);

    wire [31:0] data_idx = addr_0;
    wire in_bounds = bounds_0;

    // ================================================================
    //  BRAM read address — driven combinationally
    //  Parent BRAM will latch this on the next posedge and output
    //  fm_rd_data one cycle later.
    // ================================================================
    assign fm_rd_addr = data_idx;

    // ================================================================
    //  Delayed in_bounds — matches BRAM read latency
    //  fm_rd_data arrives 1 cycle after fm_rd_addr is set.
    //  in_bounds_d tells us whether that data is valid.
    // ================================================================
    reg in_bounds_d;
    always @(posedge clk) begin
        if (!rstn)
            in_bounds_d <= 1'b0;
        else
            in_bounds_d <= in_bounds;
    end

    // ================================================================
    //  Input data mux — uses BRAM output directly (no re-register)
    //  fm_rd_data and p1_code are both registered outputs
    //  available on the same clock edge — pipeline is preserved.
    // ================================================================
    wire signed [BITS:0] p1_data_mux;
    assign p1_data_mux = in_bounds_d ? fm_rd_data : {(BITS+1){1'b0}};

    wire w_zero_0;
    wire w_zero_1;
    wire w_zero_2;
    wire w_zero_3;

    wire mask_0 = bounds_0 ? act_mask[addr_0] : 1'b0;
    wire mask_1 = bounds_1 ? act_mask[addr_1] : 1'b0;
    wire mask_2 = bounds_2 ? act_mask[addr_2] : 1'b0;
    wire mask_3 = bounds_3 ? act_mask[addr_3] : 1'b0;

    wire skip_0 = (!mask_0) || w_zero_0;
    wire skip_1 = (!mask_1) || w_zero_1;
    wire skip_2 = (!mask_2) || w_zero_2;
    wire skip_3 = (!mask_3) || w_zero_3;

    // Compute advance amount (1 to 4)
    wire [2:0] skip_advance;
    assign skip_advance = (!skip_1) ? 3'd1 :
                          (eoc_1)   ? 3'd1 :
                          (!skip_2) ? 3'd2 :
                          (eoc_2)   ? 3'd2 :
                          (!skip_3) ? 3'd3 :
                          (eoc_3)   ? 3'd3 : 3'd4;

    wire transition_to_drain = (skip_advance == 3'd1 && eoc_1) ||
                               (skip_advance == 3'd2 && eoc_2) ||
                               (skip_advance == 3'd3 && eoc_3) ||
                               (skip_advance == 3'd4 && (eoc_4 || (ch_4 == 0 && kr_4 == 0 && kc_4 == 0)));

    reg skip_0_d;
    always @(posedge clk) begin
        if (!rstn)
            skip_0_d <= 1'b1;
        else
            skip_0_d <= skip_0;
    end

    wire signed [BITS:0] abs_input = (p1_data_mux < 0) ? -p1_data_mux : p1_data_mux;
    wire skip_this_tap = skip_0_d;

    // Per-filter: registered 2-bit ternary code
    reg signed [1:0] p1_code [0 : PARALLEL_CH - 1];

    generate
        if (PARALLEL_CH == 4) begin : gen_par_4
            // Separate 1D arrays to bypass Vivado's 1,000,000 bit limit
            // on 2D arrays during elaboration.
            // 2-bit ternary codes use distributed RAM (no BRAM needed).
            (* ram_style = "distributed" *) reg signed [1:0] w_rom_split_0 [0 : SUB_W_SIZE - 1];
            (* ram_style = "distributed" *) reg signed [1:0] w_rom_split_1 [0 : SUB_W_SIZE - 1];
            (* ram_style = "distributed" *) reg signed [1:0] w_rom_split_2 [0 : SUB_W_SIZE - 1];
            (* ram_style = "distributed" *) reg signed [1:0] w_rom_split_3 [0 : SUB_W_SIZE - 1];

            reg signed [1:0] w_rom_flat [0 : TOTAL_W - 1];

            integer wi;
            initial begin
                $readmemh(WEIGHT_FILE, w_rom_flat);
                for (wi = 0; wi < SUB_W_SIZE; wi = wi + 1) begin
                    w_rom_split_0[wi] = w_rom_flat[((wi / TAP_COUNT) * 4 + 0) * TAP_COUNT + (wi % TAP_COUNT)];
                    w_rom_split_1[wi] = w_rom_flat[((wi / TAP_COUNT) * 4 + 1) * TAP_COUNT + (wi % TAP_COUNT)];
                    w_rom_split_2[wi] = w_rom_flat[((wi / TAP_COUNT) * 4 + 2) * TAP_COUNT + (wi % TAP_COUNT)];
                    w_rom_split_3[wi] = w_rom_flat[((wi / TAP_COUNT) * 4 + 3) * TAP_COUNT + (wi % TAP_COUNT)];
                end
            end

            assign w_zero_0 = (w_rom_split_0[w_addr_0] == 2'b00) &&
                              (w_rom_split_1[w_addr_0] == 2'b00) &&
                              (w_rom_split_2[w_addr_0] == 2'b00) &&
                              (w_rom_split_3[w_addr_0] == 2'b00);

            assign w_zero_1 = (w_rom_split_0[w_addr_1] == 2'b00) &&
                              (w_rom_split_1[w_addr_1] == 2'b00) &&
                              (w_rom_split_2[w_addr_1] == 2'b00) &&
                              (w_rom_split_3[w_addr_1] == 2'b00);

            assign w_zero_2 = (w_rom_split_0[w_addr_2] == 2'b00) &&
                              (w_rom_split_1[w_addr_2] == 2'b00) &&
                              (w_rom_split_2[w_addr_2] == 2'b00) &&
                              (w_rom_split_3[w_addr_2] == 2'b00);

            assign w_zero_3 = (w_rom_split_0[w_addr_3] == 2'b00) &&
                              (w_rom_split_1[w_addr_3] == 2'b00) &&
                              (w_rom_split_2[w_addr_3] == 2'b00) &&
                              (w_rom_split_3[w_addr_3] == 2'b00);

            genvar gf;
            for (gf = 0; gf < 4; gf = gf + 1) begin : gen_filter_pipe
                always @(posedge clk) begin
                    if (gf == 0) p1_code[0] <= w_rom_split_0[group_idx * TAP_COUNT + tap_idx];
                    else if (gf == 1) p1_code[1] <= w_rom_split_1[group_idx * TAP_COUNT + tap_idx];
                    else if (gf == 2) p1_code[2] <= w_rom_split_2[group_idx * TAP_COUNT + tap_idx];
                    else if (gf == 3) p1_code[3] <= w_rom_split_3[group_idx * TAP_COUNT + tap_idx];
                end
            end
        end else begin : gen_par_generic
            // Fallback to monolithic ROM if PARALLEL_CH is not 4
            (* ram_style = "distributed" *) reg signed [1:0] w_rom [0 : TOTAL_W - 1];
            initial $readmemh(WEIGHT_FILE, w_rom);

            assign w_zero_0 = 1'b0;
            assign w_zero_1 = 1'b0;
            assign w_zero_2 = 1'b0;
            assign w_zero_3 = 1'b0;

            genvar gf;
            for (gf = 0; gf < PARALLEL_CH; gf = gf + 1) begin : gen_filter_pipe
                always @(posedge clk) begin
                    p1_code[gf] <= w_rom[(group_base + gf) * TAP_COUNT + tap_idx];
                end
            end
        end
    endgenerate

    // Pipeline validity — 1-stage
    wire feeding;
    assign feeding = (state == S_CONV_COMPUTE);

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn)
            pipe_s1_valid <= 1'b0;
        else
            pipe_s1_valid <= feeding;
    end

    // ================================================================
    //  Split accumulators — TTQ pattern
    //    pos_acc: sum of activations at +1 tap positions
    //    neg_acc: sum of activations at -1 tap positions
    //    Both BITS+24 wide to prevent overflow during accumulation.
    // ================================================================
    reg signed [BITS+24:0] pos_acc [0 : PARALLEL_CH - 1];
    reg signed [BITS+24:0] neg_acc [0 : PARALLEL_CH - 1];

    // Q16.16 extraction of pos_acc and neg_acc before Wp/Wn multiply.
    wire signed [31:0] pos_acc_q16 [0 : PARALLEL_CH - 1];
    wire signed [31:0] neg_acc_q16 [0 : PARALLEL_CH - 1];
    genvar gq;
    generate
        for (gq = 0; gq < PARALLEL_CH; gq = gq + 1) begin : gen_q16
            assign pos_acc_q16[gq] = pos_acc[gq][31:0];
            assign neg_acc_q16[gq] = neg_acc[gq][31:0];
        end
    endgenerate

    // ================================================================
    //  BN registers
    // ================================================================
    reg signed [BITS+24:0] biased_reg;
    reg signed [BITS+24:0] bn_product_reg;

    wire signed [31:0] biased_q16;
    assign biased_q16 = biased_reg[31:0];

    // Global filter index for BN/Store loop
    wire [31:0] global_store_filt;
    assign global_store_filt = group_base + store_cnt;

    // BN result (combinational)
    wire signed [BITS+24:0] bn_result;
    assign bn_result = bn_product_reg + $signed(bnsh_rom[global_store_filt]);

    // Drain counter
    reg [1:0] drain_cnt;

    // ================================================================
    //  Pool read address
    // ================================================================
    wire [31:0] pool_in_row;
    wire [31:0] pool_in_col;
    assign pool_in_row = pool_out_row * POOL_H + pool_r_cnt;
    assign pool_in_col = pool_out_col * POOL_W + pool_c_cnt;

    wire [31:0] pool_read_addr;
    assign pool_read_addr = filter_idx * CONV_POSITIONS
                          + pool_in_row * CONV_OUT_W + pool_in_col;

    reg signed [BITS:0] cur_max;

    // ================================================================
    //  Main state machine
    // ================================================================
    integer i;
    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            group_idx    <= 0;
            conv_out_row <= 0;
            conv_out_col <= 0;
            conv_pos     <= 0;
            ch_cnt       <= 0;
            kr_cnt       <= 0;
            kc_cnt       <= 0;
            store_cnt    <= 0;
            filter_idx   <= 0;
            pool_out_row <= 0;
            pool_out_col <= 0;
            pool_pos     <= 0;
            pool_r_cnt   <= 0;
            pool_c_cnt   <= 0;
            pool_counter <= 0;
            for (i = 0; i < PARALLEL_CH; i = i + 1) begin
                pos_acc[i] <= 0;
                neg_acc[i] <= 0;
            end
            biased_reg     <= 0;
            bn_product_reg <= 0;
            cur_max        <= {1'b1, {BITS{1'b0}}};
            done           <= 0;
            drain_cnt      <= 0;
            fm_wr_addr     <= 0;
            fm_wr_data     <= 0;
            fm_wr_en       <= 0;
        end else begin
            done     <= 0;
            fm_wr_en <= 0;  // Default: no write

            case (state)

                // ==================================================
                S_IDLE: begin
                    group_idx    <= 0;
                    conv_out_row <= 0;
                    conv_out_col <= 0;
                    conv_pos     <= 0;
                    ch_cnt       <= 0;
                    kr_cnt       <= 0;
                    kc_cnt       <= 0;
                    store_cnt    <= 0;
                    for (i = 0; i < PARALLEL_CH; i = i + 1) begin
                        pos_acc[i] <= 0;
                        neg_acc[i] <= 0;
                    end
                    state        <= S_CONV_COMPUTE;
                end

                // ==================================================
                //  CONV COMPUTE: PARALLEL_CH filters accumulate
                //  simultaneously — multiplier-free ternary logic.
                //  Skip tap when skip_this_tap (hysteresis or DAAP).
                // ==================================================
                S_CONV_COMPUTE: begin
                    if (pipe_s1_valid && !skip_this_tap) begin
                        for (i = 0; i < PARALLEL_CH; i = i + 1) begin
                            case (p1_code[i])
                                2'b01:   pos_acc[i] <= pos_acc[i] + p1_data_mux;  // +1
                                2'b11:   neg_acc[i] <= neg_acc[i] + p1_data_mux;  // -1
                                default: ;                                         //  0
                            endcase
                        end
                    end

                    if (transition_to_drain) begin
                        ch_cnt    <= 0;
                        kr_cnt    <= 0;
                        kc_cnt    <= 0;
                        drain_cnt <= 0;
                        state     <= S_CONV_DRAIN;
                    end else begin
                        case (skip_advance)
                            3'd1: begin
                                ch_cnt <= ch_1;
                                kr_cnt <= kr_1;
                                kc_cnt <= kc_1;
                            end
                            3'd2: begin
                                ch_cnt <= ch_2;
                                kr_cnt <= kr_2;
                                kc_cnt <= kc_2;
                            end
                            3'd3: begin
                                ch_cnt <= ch_3;
                                kr_cnt <= kr_3;
                                kc_cnt <= kc_3;
                            end
                            3'd4: begin
                                ch_cnt <= ch_4;
                                kr_cnt <= kr_4;
                                kc_cnt <= kc_4;
                            end
                            default: ;
                        endcase
                    end
                end

                // ==================================================
                //  CONV DRAIN: Flush 1-stage pipeline (2 cycles).
                //   drain_cnt==0: collect last p1 tap into pos/neg acc
                //   drain_cnt==1: accumulators final, move to SCALE
                // ==================================================
                S_CONV_DRAIN: begin
                    if (drain_cnt == 2'd0) begin
                        if (pipe_s1_valid && !skip_this_tap) begin
                            for (i = 0; i < PARALLEL_CH; i = i + 1) begin
                                case (p1_code[i])
                                    2'b01:   pos_acc[i] <= pos_acc[i] + p1_data_mux;  // +1
                                    2'b11:   neg_acc[i] <= neg_acc[i] + p1_data_mux;  // -1
                                    default: ;                                         //  0
                                endcase
                            end
                        end
                    end
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1) begin
                        store_cnt <= 0;
                        state     <= S_CONV_SCALE;
                    end
                end

                // ==================================================
                //  CONV SCALE — TTQ Wp/Wn multiply + bias (NEW STATE)
                // ==================================================
                S_CONV_SCALE: begin
                    biased_reg <= (($signed(wp_val) * pos_acc_q16[store_cnt]) >>> 16)
                                - (($signed(wn_val) * neg_acc_q16[store_cnt]) >>> 16)
                                + $signed(b_rom[global_store_filt]);
                    state <= S_CONV_BN;
                end

                // ==================================================
                //  CONV BN: Folded BatchNorm multiply (1 cycle)
                // ==================================================
                S_CONV_BN: begin
                    bn_product_reg <= ($signed(bns_rom[global_store_filt]) * biased_q16) >>> 16;
                    state          <= S_CONV_STORE;
                end

                // ==================================================
                //  CONV STORE: BN shift + ReLU + store (1 cycle)
                // ==================================================
                S_CONV_STORE: begin
                    if (activation_function) begin
                        if (bn_result > 0) begin
                            if (HAS_POOL)
                                conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= bn_result[BITS:0];
                            else begin
                                fm_wr_addr <= global_store_filt * CONV_POSITIONS + conv_pos;
                                fm_wr_data <= bn_result[BITS:0];
                                fm_wr_en   <= 1;
                            end
                        end else begin
                            if (HAS_POOL)
                                conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= 0;
                            else begin
                                fm_wr_addr <= global_store_filt * CONV_POSITIONS + conv_pos;
                                fm_wr_data <= 0;
                                fm_wr_en   <= 1;
                            end
                        end
                    end else begin
                        if (HAS_POOL)
                            conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= bn_result[BITS:0];
                        else begin
                            fm_wr_addr <= global_store_filt * CONV_POSITIONS + conv_pos;
                            fm_wr_data <= bn_result[BITS:0];
                            fm_wr_en   <= 1;
                        end
                    end

                    if (store_cnt < PARALLEL_CH - 1) begin
                        store_cnt <= store_cnt + 1;
                        state     <= S_CONV_SCALE;
                    end else begin
                        for (i = 0; i < PARALLEL_CH; i = i + 1) begin
                            pos_acc[i] <= 0;
                            neg_acc[i] <= 0;
                        end

                        if (conv_pos == CONV_POSITIONS - 1) begin
                            if (HAS_POOL) begin
                                filter_idx   <= 0;
                                pool_out_row <= 0;
                                pool_out_col <= 0;
                                pool_pos     <= 0;
                                pool_r_cnt   <= 0;
                                pool_c_cnt   <= 0;
                                pool_counter <= 0;
                                cur_max      <= {1'b1, {BITS{1'b0}}};
                                state        <= S_POOL_COMPARE;
                            end else begin
                                if (group_idx < NUM_GROUPS - 1) begin
                                    group_idx    <= group_idx + 1;
                                    conv_out_row <= 0;
                                    conv_out_col <= 0;
                                    conv_pos     <= 0;
                                    ch_cnt       <= 0;
                                    kr_cnt       <= 0;
                                    kc_cnt       <= 0;
                                    state        <= S_CONV_COMPUTE;
                                end else
                                    state <= S_DONE;
                            end
                        end else begin
                            conv_pos <= conv_pos + 1;
                            if (conv_out_col == CONV_OUT_W - 1) begin
                                conv_out_col <= 0;
                                conv_out_row <= conv_out_row + 1;
                            end else
                                conv_out_col <= conv_out_col + 1;
                            state <= S_CONV_COMPUTE;
                        end
                    end
                end

                // ==================================================
                //  POOL COMPARE — Max over POOL_H × POOL_W window
                // ==================================================
                S_POOL_COMPARE: begin
                    if ($signed(conv_buf[pool_read_addr]) > $signed(cur_max))
                        cur_max <= conv_buf[pool_read_addr];

                    if (pool_counter == POOL_ELEMENTS - 1) begin
                        pool_counter <= 0;
                        pool_r_cnt   <= 0;
                        pool_c_cnt   <= 0;
                        state        <= S_POOL_STORE;
                    end else begin
                        pool_counter <= pool_counter + 1;
                        if (pool_c_cnt == POOL_W - 1) begin
                            pool_c_cnt <= 0;
                            pool_r_cnt <= pool_r_cnt + 1;
                        end else
                            pool_c_cnt <= pool_c_cnt + 1;
                    end
                end

                // ==================================================
                //  POOL STORE — Write max to parent BRAM
                // ==================================================
                S_POOL_STORE: begin
                    fm_wr_addr <= (group_base + filter_idx) * POOL_OUT_POS + pool_pos;
                    fm_wr_data <= cur_max;
                    fm_wr_en   <= 1;
                    cur_max    <= {1'b1, {BITS{1'b0}}};

                    if (pool_pos == POOL_OUT_POS - 1) begin
                        if (filter_idx == PARALLEL_CH - 1) begin
                            if (group_idx < NUM_GROUPS - 1) begin
                                group_idx    <= group_idx + 1;
                                conv_out_row <= 0;
                                conv_out_col <= 0;
                                conv_pos     <= 0;
                                ch_cnt       <= 0;
                                kr_cnt       <= 0;
                                kc_cnt       <= 0;
                                state        <= S_CONV_COMPUTE;
                            end else
                                state <= S_DONE;
                        end else begin
                            filter_idx   <= filter_idx + 1;
                            pool_out_row <= 0;
                            pool_out_col <= 0;
                            pool_pos     <= 0;
                            pool_r_cnt   <= 0;
                            pool_c_cnt   <= 0;
                            pool_counter <= 0;
                            state        <= S_POOL_COMPARE;
                        end
                    end else begin
                        pool_pos <= pool_pos + 1;
                        if (pool_out_col == POOL_OUT_W - 1) begin
                            pool_out_col <= 0;
                            pool_out_row <= pool_out_row + 1;
                        end else
                            pool_out_col <= pool_out_col + 1;
                        state <= S_POOL_COMPARE;
                    end
                end

                // ==================================================
                S_DONE: begin
                    done <= 1;
                end

            endcase
        end
    end

endmodule
