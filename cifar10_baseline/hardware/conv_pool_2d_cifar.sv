`timescale 1ns / 1ps
//============================================================================
// Merged Conv2D + BN + ReLU + MaxPool2D — Parallel-filter group processing
//
// CIFAR-10 Baseline: combines parallel computation (from hardware_parallel)
// with BatchNorm (from hardware_sequential_bn), adding:
//   • Zero-padding support (PAD_SIZE parameter)
//   • Group parallelism (PARALLEL_CH filters computed simultaneously)
//   • Optional pooling (HAS_POOL parameter — Conv3 has no pool)
//
// Group processing:
//   OUT_CH filters are divided into NUM_GROUPS = OUT_CH / PARALLEL_CH groups.
//   Each group processes PARALLEL_CH filters simultaneously using PARALLEL_CH
//   multipliers. Groups are iterated sequentially.
//
// Padding:
//   PAD_SIZE=1 (same-padding for 3×3 kernel) preserves spatial dimensions:
//     CONV_OUT_H = IN_H + 2*PAD_SIZE - KERNEL_H + 1 = IN_H  (when pad=1, k=3)
//   Out-of-bounds positions read zero instead of data_in.
//
// BN (always included):
//   After drain, each filter's accumulated value is bias-added, then:
//     bn_product = (bn_scale × biased_q16) >>> 16
//     bn_result  = bn_product + bn_shift
//   Then ReLU applied. Uses 1 DSP48 (time-shared across store_cnt).
//
// State flow per group:
//   S_IDLE → S_CONV_COMPUTE (PARALLEL_CH multipliers) → S_CONV_DRAIN (2 cyc)
//   → S_CONV_BN / S_CONV_STORE loop (2 cycles × PARALLEL_CH filters)
//   → [next position → S_CONV_COMPUTE]
//   → [all positions, HAS_POOL=1 → S_POOL_COMPARE → S_POOL_STORE]
//   → [all positions, HAS_POOL=0, more groups → S_CONV_COMPUTE]
//   → [all done → S_DONE]
//
// Conv1: IN=32×32×3,  OUT_CH=32, PARALLEL_CH=16 → 2 groups, HAS_POOL=1
//   TAP=27, per-pos=27+2+32=61 cyc, conv=1024×61×2=124,928, pool=2×16×256×5=40,960
//   Total ≈ 165,888 cycles
// Conv2: IN=16×16×32, OUT_CH=64, PARALLEL_CH=16 → 4 groups, HAS_POOL=1
//   TAP=288, per-pos=288+2+32=322 cyc, conv=256×322×4=329,728, pool=4×16×64×5=20,480
//   Total ≈ 350,208 cycles
// Conv3: IN=8×8×64,   OUT_CH=64, PARALLEL_CH=16 → 4 groups, HAS_POOL=0
//   TAP=576, per-pos=576+2+32=610 cyc, conv=64×610×4=156,160
//   Total ≈ 156,160 cycles
//
// Fixed-point: Q16.16
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
module conv_pool_2d_cifar #(
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
    parameter PARALLEL_CH = 16,
    parameter CONV_OUT_H  = IN_H + 2*PAD_SIZE - KERNEL_H + 1,
    parameter CONV_OUT_W  = IN_W + 2*PAD_SIZE - KERNEL_W + 1,
    parameter POOL_OUT_H  = CONV_OUT_H / POOL_H,
    parameter POOL_OUT_W  = CONV_OUT_W / POOL_W,
    parameter BITS        = 31
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     activation_function,

    input  wire signed [BITS:0]     data_in  [0 : IN_H * IN_W * IN_CH - 1],
    input  wire signed [31:0]       weights  [0 : OUT_CH * IN_CH * KERNEL_H * KERNEL_W - 1],
    input  wire signed [31:0]       bias     [0 : OUT_CH - 1],

    // Folded BN parameters (Q16.16 per output channel)
    input  wire signed [31:0]       bn_scale [0 : OUT_CH - 1],
    input  wire signed [31:0]       bn_shift [0 : OUT_CH - 1],

    output reg  signed [BITS:0]     data_out [0 : OUT_CH * (HAS_POOL ? (POOL_OUT_H * POOL_OUT_W) : (CONV_OUT_H * CONV_OUT_W)) - 1],
    output reg                      done
);

    // ================================================================
    //  Constants
    // ================================================================
    localparam TAP_COUNT      = IN_CH * KERNEL_H * KERNEL_W;
    localparam CONV_POSITIONS = CONV_OUT_H * CONV_OUT_W;
    localparam POOL_OUT_POS   = POOL_OUT_H * POOL_OUT_W;
    localparam POOL_ELEMENTS  = POOL_H * POOL_W;
    localparam NUM_GROUPS     = OUT_CH / PARALLEL_CH;
    localparam OUT_POSITIONS  = HAS_POOL ? POOL_OUT_POS : CONV_POSITIONS;

    // ================================================================
    //  Conv buffer — PARALLEL_CH filters (reused per group)
    //  Conv1: 16 × 1024 × 32 = 524,288 bits → ~15 BRAM36
    //  Conv2: 16 × 256  × 32 = 131,072 bits → ~4  BRAM36
    //  Conv3: not used (HAS_POOL=0, write direct to data_out)
    // ================================================================
    (* ram_style = "block" *) reg signed [BITS:0] conv_buf [0 : PARALLEL_CH * CONV_POSITIONS - 1];

    // ================================================================
    //  State machine — 4 bits (8 states)
    // ================================================================
    localparam S_IDLE         = 4'd0;
    localparam S_CONV_COMPUTE = 4'd1;
    localparam S_CONV_DRAIN   = 4'd2;
    localparam S_CONV_BN      = 4'd3;
    localparam S_CONV_STORE   = 4'd4;
    localparam S_POOL_COMPARE = 4'd5;
    localparam S_POOL_STORE   = 4'd6;
    localparam S_DONE         = 4'd7;

    reg [3:0] state;

    // ================================================================
    //  Group counter
    // ================================================================
    reg [31:0] group_idx;
    wire [31:0] group_base;
    assign group_base = group_idx * PARALLEL_CH;

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
    reg [31:0] filter_idx;   // 0..PARALLEL_CH-1 within group (for pool)
    reg [31:0] pool_out_row;
    reg [31:0] pool_out_col;
    reg [31:0] pool_pos;
    reg [31:0] pool_r_cnt;
    reg [31:0] pool_c_cnt;
    reg [31:0] pool_counter;

    // ================================================================
    //  Padding — compute actual input coordinates
    //  conv_out_row + kr_cnt is the position in the padded input.
    //  Subtract PAD_SIZE to get actual input coordinate.
    //  If out of bounds, use zero.
    // ================================================================
    wire [31:0] pad_row_sum;
    wire [31:0] pad_col_sum;
    assign pad_row_sum = conv_out_row + kr_cnt;
    assign pad_col_sum = conv_out_col + kc_cnt;

    wire in_row_valid = (pad_row_sum >= PAD_SIZE) &&
                        (pad_row_sum <  PAD_SIZE + IN_H);
    wire in_col_valid = (pad_col_sum >= PAD_SIZE) &&
                        (pad_col_sum <  PAD_SIZE + IN_W);
    wire in_bounds    = in_row_valid && in_col_valid;

    wire [31:0] actual_row = pad_row_sum - PAD_SIZE;
    wire [31:0] actual_col = pad_col_sum - PAD_SIZE;
    wire [31:0] data_idx   = ch_cnt * (IN_H * IN_W)
                            + actual_row * IN_W + actual_col;

    // Tap index within one filter's weight set
    wire [31:0] tap_idx;
    assign tap_idx = ch_cnt * (KERNEL_H * KERNEL_W) + kr_cnt * KERNEL_W + kc_cnt;

    // ================================================================
    //  Parallel-filter datapath — 1-stage pipeline
    //  PARALLEL_CH multipliers active simultaneously.
    //  Shared input data, per-filter weights.
    // ================================================================

    // Stage 1: Register shared data value (with padding)
    reg signed [BITS:0] p1_data;
    always @(posedge clk) begin
        if (in_bounds)
            p1_data <= data_in[data_idx];
        else
            p1_data <= {(BITS+1){1'b0}};
    end

    // Per-filter: registered weight + combinational multiply
    reg  signed [31:0]      p1_weight      [0 : PARALLEL_CH - 1];
    wire signed [BITS+32:0] p1_full_product [0 : PARALLEL_CH - 1];
    wire signed [BITS+16:0] p1_product     [0 : PARALLEL_CH - 1];

    genvar gf;
    generate
        for (gf = 0; gf < PARALLEL_CH; gf = gf + 1) begin : gen_filter_pipe
            always @(posedge clk) begin
                p1_weight[gf] <= weights[(group_base + gf) * TAP_COUNT + tap_idx];
            end
            assign p1_full_product[gf] = p1_weight[gf] * p1_data;
            assign p1_product[gf]      = p1_full_product[gf] >>> 16;
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
    reg signed [BITS+24:0] acc [0 : PARALLEL_CH - 1];

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
    assign bn_result = bn_product_reg + $signed(bn_shift[global_store_filt]);

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
            for (i = 0; i < PARALLEL_CH; i = i + 1)
                acc[i]   <= 0;
            biased_reg     <= 0;
            bn_product_reg <= 0;
            cur_max        <= {1'b1, {BITS{1'b0}}};
            done           <= 0;
            drain_cnt      <= 0;
        end else begin
            done <= 0;

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
                    for (i = 0; i < PARALLEL_CH; i = i + 1)
                        acc[i]   <= 0;
                    state        <= S_CONV_COMPUTE;
                end

                // ==================================================
                //  CONV COMPUTE: PARALLEL_CH filters accumulate
                //  simultaneously (PARALLEL_CH multipliers in parallel).
                // ==================================================
                S_CONV_COMPUTE: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < PARALLEL_CH; i = i + 1)
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
                //  CONV DRAIN: Flush 1-stage pipeline (2 cycles).
                //  At drain_cnt==1: acc is final, register biased
                //  for filter 0 in group, start BN loop.
                // ==================================================
                S_CONV_DRAIN: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < PARALLEL_CH; i = i + 1)
                            acc[i] <= acc[i] + p1_product[i];
                    end
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1) begin
                        store_cnt  <= 0;
                        biased_reg <= acc[0] + $signed(bias[group_base]);
                        state      <= S_CONV_BN;
                    end
                end

                // ==================================================
                //  CONV BN: Folded BatchNorm multiply (1 cycle)
                //  Processes one filter at a time via store_cnt.
                //  bn_product = (bn_scale × biased_q16) >>> 16
                // ==================================================
                S_CONV_BN: begin
                    bn_product_reg <= ($signed(bn_scale[global_store_filt]) * biased_q16) >>> 16;
                    state          <= S_CONV_STORE;
                end

                // ==================================================
                //  CONV STORE: BN shift + ReLU + store (1 cycle)
                //  If HAS_POOL: store to conv_buf
                //  If !HAS_POOL: store directly to data_out
                //  Then advance to next filter or next position.
                // ==================================================
                S_CONV_STORE: begin
                    // BN shift + ReLU + store
                    if (activation_function) begin
                        if (bn_result > 0) begin
                            if (HAS_POOL)
                                conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= bn_result[BITS:0];
                            else
                                data_out[global_store_filt * CONV_POSITIONS + conv_pos] <= bn_result[BITS:0];
                        end else begin
                            if (HAS_POOL)
                                conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= 0;
                            else
                                data_out[global_store_filt * CONV_POSITIONS + conv_pos] <= 0;
                        end
                    end else begin
                        if (HAS_POOL)
                            conv_buf[store_cnt * CONV_POSITIONS + conv_pos] <= bn_result[BITS:0];
                        else
                            data_out[global_store_filt * CONV_POSITIONS + conv_pos] <= bn_result[BITS:0];
                    end

                    if (store_cnt < PARALLEL_CH - 1) begin
                        // Next filter in group — register biased for next
                        biased_reg <= acc[store_cnt + 1]
                                    + $signed(bias[group_base + store_cnt + 1]);
                        store_cnt  <= store_cnt + 1;
                        state      <= S_CONV_BN;
                    end else begin
                        // All PARALLEL_CH filters stored for this position
                        for (i = 0; i < PARALLEL_CH; i = i + 1)
                            acc[i] <= 0;

                        if (conv_pos == CONV_POSITIONS - 1) begin
                            // All conv positions done for this group
                            if (HAS_POOL) begin
                                // Start pooling for this group
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
                                // No pool — check for more groups
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
                            // Next conv output position
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
                //  Processes one filter at a time (filter_idx within
                //  the current group, 0..PARALLEL_CH-1).
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
                //  POOL STORE — Write max to data_out at global index
                // ==================================================
                S_POOL_STORE: begin
                    data_out[(group_base + filter_idx) * POOL_OUT_POS + pool_pos] <= cur_max;
                    cur_max <= {1'b1, {BITS{1'b0}}};

                    if (pool_pos == POOL_OUT_POS - 1) begin
                        if (filter_idx == PARALLEL_CH - 1) begin
                            // All filters in this group pooled
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
