`timescale 1ns / 1ps
//============================================================================
// Merged Conv2D + MaxPool2D — Parallel-filter processing
//
// All output filters computed simultaneously during convolution.
// Pool phase processes filters sequentially (single conv_buf read port).
//
// This is the NATURAL full-precision architecture: since every tap requires
// a real 32×32 multiply (unlike TTQ which uses add/subtract), the baseline
// instantiates OUT_CH multipliers to process all filters in parallel.
// This makes the DSP cost of full-precision weights visible in synthesis.
//
// Weight storage:
//   Conv weights passed as port arrays from cnn2d_synth_top (LUT-ROM).
//   OUT_CH weight MUX trees read simultaneously → large LUT footprint.
//   Conv buffer (all filters) stored in BRAM via (* ram_style = "block" *).
//
// State flow:
//   S_CONV_COMPUTE → S_CONV_DRAIN → S_CONV_STORE (OUT_CH cycles)
//   → … (all positions) → S_POOL_COMPARE → S_POOL_STORE → S_DONE
//
// 1-stage pipeline (matches TTQ architecture):
//   Stage 1 (posedge): address calc + array MUX → registered data/weights
//   Accumulate (posedge): OUT_CH combinational multiplies + accumulate
//   S_CONV_DRAIN flushes the 1-stage pipeline after all taps are issued.
//   S_CONV_STORE writes one filter per cycle (BRAM single-port compatible).
//
// Cycle count:
//   Conv phase:  CONV_POSITIONS × (TAP_COUNT + 2 + OUT_CH)
//   Pool phase:  OUT_CH × POOL_OUT_POS × (POOL_ELEMENTS + 1)
//   Conv1: 676×(9+2+4) + 4×169×5 = 10140+3380 = 13,520 cycles
//   Conv2: 121×(36+2+8) + 8×25×5 = 5566+1000  =  6,566 cycles
//   Total conv: ~20,086  Total with FC: ~28,700
//
// Resource cost vs TTQ:
//   DSPs: OUT_CH multipliers (4+8=12 × ~4 DSP48 ≈ 48-67 DSPs)
//         TTQ uses 0 DSPs per tap; pays ~12 DSPs per module for
//         Wp/Wn/BN multiplies = ~51 total
//   LUTs: OUT_CH weight MUX trees + large conv_buf decode logic
//   BRAM: ~4 BRAM36 for conv_buf (86K+31K bits total)
//
// Fixed-point: Q16.16
//============================================================================
module conv_pool_2d_pl #(
    parameter IN_H       = 28,
    parameter IN_W       = 28,
    parameter IN_CH      = 1,
    parameter OUT_CH     = 4,
    parameter KERNEL_H   = 3,
    parameter KERNEL_W   = 3,
    parameter POOL_H     = 2,
    parameter POOL_W     = 2,
    parameter CONV_OUT_H = IN_H - KERNEL_H + 1,
    parameter CONV_OUT_W = IN_W - KERNEL_W + 1,
    parameter POOL_OUT_H = CONV_OUT_H / POOL_H,
    parameter POOL_OUT_W = CONV_OUT_W / POOL_W,
    parameter BITS       = 31
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     activation_function,

    input  wire signed [BITS:0]     data_in [0 : IN_H * IN_W * IN_CH - 1],
    input  wire signed [31:0]       weights [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],
    input  wire signed [31:0]       bias    [0 : OUT_CH - 1],

    output reg  signed [BITS:0]     data_out [0 : POOL_OUT_H * POOL_OUT_W * OUT_CH - 1],
    output reg                      done
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam TAP_COUNT      = IN_CH * KERNEL_H * KERNEL_W;
    localparam CONV_POSITIONS = CONV_OUT_H * CONV_OUT_W;
    localparam POOL_OUT_POS   = POOL_OUT_H * POOL_OUT_W;
    localparam POOL_ELEMENTS  = POOL_H * POOL_W;

    // ================================================================
    //  Conv buffer — ALL filters stored (BRAM-inferred)
    //  Conv1: 4 × 676 × 32 = 86,528 bits → ~3 BRAM36
    //  Conv2: 8 × 121 × 32 = 30,976 bits → ~1 BRAM36
    //  Written one filter per cycle (single write port → BRAM-compatible)
    // ================================================================
    (* ram_style = "block" *) reg signed [BITS:0] conv_buf [0 : OUT_CH * CONV_POSITIONS - 1];

    // ================================================================
    //  State machine
    // ================================================================
    localparam S_IDLE         = 3'd0;
    localparam S_CONV_COMPUTE = 3'd1;
    localparam S_CONV_DRAIN   = 3'd2;
    localparam S_CONV_STORE   = 3'd3;  // OUT_CH cycles (one filter per cycle)
    localparam S_POOL_COMPARE = 3'd4;
    localparam S_POOL_STORE   = 3'd5;
    localparam S_DONE         = 3'd6;

    reg [2:0] state;

    // ================================================================
    //  Conv counters (NO filter_idx loop — all filters parallel in compute)
    // ================================================================
    reg [31:0] conv_out_row;
    reg [31:0] conv_out_col;
    reg [31:0] conv_pos;
    reg [31:0] ch_cnt;
    reg [31:0] kr_cnt;
    reg [31:0] kc_cnt;

    // ================================================================
    //  Pool counters (filter_idx used during pool + sequential store)
    // ================================================================
    reg [31:0] filter_idx;
    reg [31:0] pool_out_row;
    reg [31:0] pool_out_col;
    reg [31:0] pool_pos;
    reg [31:0] pool_r_cnt;
    reg [31:0] pool_c_cnt;
    reg [31:0] pool_counter;

    // Store counter — sequential BRAM write (one filter per cycle)
    reg [31:0] store_cnt;

    // ================================================================
    //  Parallel-filter datapath — 1-stage pipeline
    //
    //  All OUT_CH filters share the same input data (one read per cycle).
    //  Each filter has its own weight register, multiplier, accumulator.
    //  OUT_CH × (32×32) multipliers = natural DSP cost of full-precision.
    // ================================================================

    // Shared address computation
    wire [31:0] data_idx;
    assign data_idx = ch_cnt * (IN_H * IN_W)
                    + (conv_out_row + kr_cnt) * IN_W
                    + (conv_out_col + kc_cnt);

    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * (KERNEL_H * KERNEL_W) + kr_cnt * KERNEL_W + kc_cnt;

    // Stage 1: Register shared data value
    reg signed [BITS:0] p1_data;
    always @(posedge clk) begin
        p1_data <= data_in[data_idx];
    end

    // Per-filter: registered weight + combinational multiply
    reg  signed [31:0]      p1_weight [0 : OUT_CH - 1];
    wire signed [BITS+32:0] p1_full_product [0 : OUT_CH - 1];
    wire signed [BITS+16:0] p1_product [0 : OUT_CH - 1];

    genvar f;
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_filter_pipe
            always @(posedge clk) begin
                p1_weight[f] <= weights[f * TAP_COUNT + tap_idx];
            end
            assign p1_full_product[f] = p1_weight[f] * p1_data;
            assign p1_product[f] = p1_full_product[f] >>> 16;
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

    // Per-filter accumulators
    reg signed [BITS+24:0] acc [0 : OUT_CH - 1];

    // Per-filter biased values (combinational)
    wire signed [BITS+24:0] biased [0 : OUT_CH - 1];
    generate
        for (f = 0; f < OUT_CH; f = f + 1) begin : gen_biased
            assign biased[f] = acc[f] + bias[f];
        end
    endgenerate

    // Drain counter
    reg [1:0] drain_cnt;

    // Pool read address
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
            conv_out_row <= 0;
            conv_out_col <= 0;
            conv_pos     <= 0;
            ch_cnt       <= 0;
            kr_cnt       <= 0;
            kc_cnt       <= 0;
            filter_idx   <= 0;
            pool_out_row <= 0;
            pool_out_col <= 0;
            pool_pos     <= 0;
            pool_r_cnt   <= 0;
            pool_c_cnt   <= 0;
            pool_counter <= 0;
            store_cnt    <= 0;
            for (i = 0; i < OUT_CH; i = i + 1)
                acc[i]   <= 0;
            cur_max      <= {1'b1, {BITS{1'b0}}};
            done         <= 0;
            drain_cnt    <= 0;
        end else begin
            done <= 0;

            case (state)

                // ==================================================
                S_IDLE: begin
                    conv_out_row <= 0;
                    conv_out_col <= 0;
                    conv_pos     <= 0;
                    ch_cnt       <= 0;
                    kr_cnt       <= 0;
                    kc_cnt       <= 0;
                    store_cnt    <= 0;
                    for (i = 0; i < OUT_CH; i = i + 1)
                        acc[i]   <= 0;
                    state        <= S_CONV_COMPUTE;
                end

                // ==================================================
                //  CONV COMPUTE: All OUT_CH filters accumulate
                //  simultaneously (OUT_CH multipliers in parallel).
                // ==================================================
                S_CONV_COMPUTE: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < OUT_CH; i = i + 1)
                            acc[i] <= acc[i] + p1_product[i];
                    end

                    if (kc_cnt == KERNEL_W - 1) begin
                        kc_cnt <= 0;
                        if (kr_cnt == KERNEL_H - 1) begin
                            kr_cnt <= 0;
                            if (ch_cnt == IN_CH - 1) begin
                                ch_cnt    <= 0;
                                drain_cnt <= 0;
                                state     <= S_CONV_DRAIN;
                            end else
                                ch_cnt <= ch_cnt + 1;
                        end else
                            kr_cnt <= kr_cnt + 1;
                    end else
                        kc_cnt <= kc_cnt + 1;
                end

                // ==================================================
                //  CONV DRAIN: Flush 1-stage pipeline (2 cycles)
                // ==================================================
                S_CONV_DRAIN: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < OUT_CH; i = i + 1)
                            acc[i] <= acc[i] + p1_product[i];
                    end
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1) begin
                        store_cnt <= 0;
                        state     <= S_CONV_STORE;
                    end
                end

                // ==================================================
                //  CONV STORE: Write one filter per cycle (BRAM-safe).
                //  Takes OUT_CH cycles per position.
                // ==================================================
                S_CONV_STORE: begin
                    if (activation_function) begin
                        if (biased[store_cnt] > 0)
                            conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= biased[store_cnt][BITS:0];
                        else
                            conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= 0;
                    end else
                        conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= biased[store_cnt][BITS:0];

                    if (store_cnt == OUT_CH - 1) begin
                        store_cnt <= 0;
                        for (i = 0; i < OUT_CH; i = i + 1)
                            acc[i] <= 0;

                        if (conv_pos == CONV_POSITIONS - 1) begin
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
                            conv_pos <= conv_pos + 1;
                            if (conv_out_col == CONV_OUT_W - 1) begin
                                conv_out_col <= 0;
                                conv_out_row <= conv_out_row + 1;
                            end else
                                conv_out_col <= conv_out_col + 1;
                            state <= S_CONV_COMPUTE;
                        end
                    end else
                        store_cnt <= store_cnt + 1;
                end

                // ==================================================
                S_POOL_COMPARE: begin
                    if (conv_buf[pool_read_addr] > cur_max)
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
                S_POOL_STORE: begin
                    data_out[filter_idx * POOL_OUT_POS + pool_pos] <= cur_max;
                    cur_max <= {1'b1, {BITS{1'b0}}};

                    if (pool_pos == POOL_OUT_POS - 1) begin
                        if (filter_idx == OUT_CH - 1) begin
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
