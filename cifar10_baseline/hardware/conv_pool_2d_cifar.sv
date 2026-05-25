`timescale 1ns / 1ps
//============================================================================
// Conv2D + BatchNorm + Optional MaxPool — CIFAR-10 (BRAM Interface)
//
// ARCHITECTURE CHANGE: Feature maps use BRAM-based address/data ports
// instead of full unpacked array ports.
//   - Input:  fm_rd_addr (output) / fm_rd_data (input) — reads from parent BRAM
//   - Output: fm_wr_addr, fm_wr_data, fm_wr_en — writes to parent BRAM
//
// This eliminates hundreds of thousands of flip-flops that were needed
// for the old unpacked array ports (e.g., 8192 × 32 = 262K FFs per layer).
//
// Pipeline timing:
//   fm_rd_addr is driven combinationally. Parent BRAM output (fm_rd_data)
//   arrives 1 cycle later — same latency as the internal w_rom BRAM read
//   for p1_weight. Both are available on the same clock edge.
//
// Fixed-point: Q16.16
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
(* KEEP_HIERARCHY = "yes" *) module conv_pool_2d_cifar #(
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
    parameter BITS        = 31,

    // Weight/BN file paths — loaded via $readmemh into internal ROMs.
    parameter WEIGHT_FILE   = "",
    parameter BIAS_FILE     = "",
    parameter BN_SCALE_FILE = "",
    parameter BN_SHIFT_FILE = ""
)(
    input  wire                     clk,
    input  wire                     rstn,
    input  wire                     activation_function,

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
    //  Internal weight/BN ROMs (file-loaded)
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
    //  State machine
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
    reg [31:0] filter_idx;
    reg [31:0] pool_out_row;
    reg [31:0] pool_out_col;
    reg [31:0] pool_pos;
    reg [31:0] pool_r_cnt;
    reg [31:0] pool_c_cnt;
    reg [31:0] pool_counter;

    // ================================================================
    //  Padding — compute actual input coordinates
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
    //  This replaces the old p1_data register.
    //  fm_rd_data and p1_weight are both registered BRAM outputs
    //  available on the same clock edge — pipeline is preserved.
    // ================================================================
    wire signed [BITS:0] p1_data_mux;
    assign p1_data_mux = in_bounds_d ? fm_rd_data : {(BITS+1){1'b0}};

    // ================================================================
    //  Parallel-filter datapath — 1-stage pipeline
    //  PARALLEL_CH multipliers active simultaneously.
    //  Shared input data, per-filter weights.
    // ================================================================

    // Per-filter: registered weight + combinational multiply
    reg  signed [31:0]      p1_weight      [0 : PARALLEL_CH - 1];
    wire signed [BITS+32:0] p1_full_product [0 : PARALLEL_CH - 1];
    wire signed [BITS+16:0] p1_product     [0 : PARALLEL_CH - 1];

    generate
        if (PARALLEL_CH == 4) begin : gen_par_4
            // We use separate 1D arrays to bypass Vivado's 1,000,000 bit limit on 2D arrays during elaboration.
            (* ram_style = "block" *) reg signed [31:0] w_rom_split_0 [0 : SUB_W_SIZE - 1];
            (* ram_style = "block" *) reg signed [31:0] w_rom_split_1 [0 : SUB_W_SIZE - 1];
            (* ram_style = "block" *) reg signed [31:0] w_rom_split_2 [0 : SUB_W_SIZE - 1];
            (* ram_style = "block" *) reg signed [31:0] w_rom_split_3 [0 : SUB_W_SIZE - 1];

            reg signed [31:0] w_rom_flat [0 : TOTAL_W - 1];

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

            genvar gf;
            for (gf = 0; gf < 4; gf = gf + 1) begin : gen_filter_pipe
                always @(posedge clk) begin
                    if (gf == 0) p1_weight[0] <= w_rom_split_0[group_idx * TAP_COUNT + tap_idx];
                    else if (gf == 1) p1_weight[1] <= w_rom_split_1[group_idx * TAP_COUNT + tap_idx];
                    else if (gf == 2) p1_weight[2] <= w_rom_split_2[group_idx * TAP_COUNT + tap_idx];
                    else if (gf == 3) p1_weight[3] <= w_rom_split_3[group_idx * TAP_COUNT + tap_idx];
                end
                (* use_dsp = "yes" *) assign p1_full_product[gf] = p1_weight[gf] * p1_data_mux;
                assign p1_product[gf]      = p1_full_product[gf] >>> 16;
            end
        end else begin : gen_par_generic
            // Fallback to monolithic ROM if PARALLEL_CH is not 4
            // (Uses distributed RAM / LUTs for storage if PARALLEL_CH > 2, but compiles successfully)
            reg signed [31:0] w_rom [0 : TOTAL_W - 1];
            initial $readmemh(WEIGHT_FILE, w_rom);

            genvar gf;
            for (gf = 0; gf < PARALLEL_CH; gf = gf + 1) begin : gen_filter_pipe
                always @(posedge clk) begin
                    p1_weight[gf] <= w_rom[(group_base + gf) * TAP_COUNT + tap_idx];
                end
                (* use_dsp = "yes" *) assign p1_full_product[gf] = p1_weight[gf] * p1_data_mux;
                assign p1_product[gf]      = p1_full_product[gf] >>> 16;
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
            for (i = 0; i < PARALLEL_CH; i = i + 1)
                acc[i]   <= 0;
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
                // ==================================================
                S_CONV_DRAIN: begin
                    if (pipe_s1_valid) begin
                        for (i = 0; i < PARALLEL_CH; i = i + 1)
                            acc[i] <= acc[i] + p1_product[i];
                    end
                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1) begin
                        store_cnt  <= 0;
                        biased_reg <= acc[0] + $signed(b_rom[group_base]);
                        state      <= S_CONV_BN;
                    end
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
                //  If HAS_POOL: store to internal conv_buf
                //  If !HAS_POOL: write to parent BRAM via fm_wr_*
                // ==================================================
                S_CONV_STORE: begin
                    // BN shift + ReLU + store
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
                        biased_reg <= acc[store_cnt + 1]
                                    + $signed(b_rom[group_base + store_cnt + 1]);
                        store_cnt  <= store_cnt + 1;
                        state      <= S_CONV_BN;
                    end else begin
                        for (i = 0; i < PARALLEL_CH; i = i + 1)
                            acc[i] <= 0;

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
                //  Reads from internal conv_buf (BRAM).
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
