`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer — TTQ + BatchNorm + DAAP Threshold (CIFAR-10)
//
// Combines:
//   • CIFAR-10 baseline's FORCE_BRAM parameter and generate blocks
//   • MNIST TTQ's split accumulator pattern (pos_acc / neg_acc)
//   • MNIST TTQ's S_SCALE state (Wp/Wn multiply + bias)
//   • DAAP activation threshold: per-neuron Q16.16 threshold ROM.
//     When |input| < cur_thresh, the tap is skipped.
//     When all thresholds are zero: never skip → pure TTQ behavior.
//
// Features:
//   • HAS_BN parameter: 1 for FC1 (BN after S_SCALE), 0 for FC2 (no BN)
//   • Sequential ternary accumulation with 1-stage pipeline
//   • 2-bit weight ROM via $readmemh (distributed or BRAM via FORCE_BRAM)
//   • PAD=20 zero-padding convention
//
// For CIFAR-10 TTQ:
//   FC1: NUM_NEURONS=256, LAYER_NEURON_WIDTH=103 (PAD+64+PAD-1)
//        HAS_BN=1, FORCE_BRAM=0
//        cycles per neuron = (103+1) + 2 + 1 + 1 + 1 = 109
//        Total: 109 × 256 = 27,904 cycles
//   FC2: NUM_NEURONS=10,  LAYER_NEURON_WIDTH=295 (PAD+256+PAD-1)
//        HAS_BN=0, FORCE_BRAM=0
//        cycles per neuron = (295+1) + 2 + 1 + 0 + 1 = 300
//        Total: 300 × 10 = 3,000 cycles
//
// FSM States (3 bits, 8 states):
//   S_IDLE → S_FILL → S_MAC → S_DRAIN → S_SCALE → S_BN → S_STORE → S_DONE
//
// Fixed-point: Q16.16
//============================================================================
(* KEEP_HIERARCHY = "yes" *) module layer_seq_cifar_ttq #(
    parameter NUM_NEURONS        = 256,
    parameter LAYER_NEURON_WIDTH = 103,    // Number of inputs − 1 (0-indexed)
    parameter LAYER_BITS         = 31,     // Input data bit width
    parameter B_BITS             = 31,     // Bias bit width
    parameter HAS_BN             = 1,      // 1 = FC1 (BN after S_SCALE), 0 = FC2
    parameter FORCE_BRAM         = 0,      // Always 0 for TTQ (2-bit weights are tiny)
    parameter WEIGHT_FILE        = "",     // Path to .mem file for $readmemh
    parameter ACT_THRESH_FILE    = ""      // Per-neuron Q16.16 threshold (NUM_NEURONS entries)
)(
    input  wire                           clk,
    input  wire                           rstn,
    input  wire                           activation_function,  // 1 = ReLU, 0 = none

    input  wire signed [B_BITS:0]         b        [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]     data_in  [0:LAYER_NEURON_WIDTH],

    // TTQ scaling factors (Q16.16 scalars — one per entire layer)
    input  wire signed [31:0]             wp,
    input  wire signed [31:0]             wn,

    // Folded BN parameters (Q16.16 per neuron, ignored when HAS_BN=0)
    input  wire signed [31:0]             bn_scale [0:NUM_NEURONS-1],
    input  wire signed [31:0]             bn_shift [0:NUM_NEURONS-1],

    output reg  signed [LAYER_BITS+8:0]   data_out [0:NUM_NEURONS-1],
    output reg                            counter_donestatus
);

    // ================================================================
    //  Weight ROM — 2-bit ternary codes, 1D flat array
    //  FORCE_BRAM=1: BRAM (kept for generate-block compatibility)
    //  FORCE_BRAM=0: distributed RAM (default for TTQ)
    // ================================================================
    localparam NUM_INPUTS    = LAYER_NEURON_WIDTH + 1;
    localparam TOTAL_WEIGHTS = NUM_NEURONS * NUM_INPUTS;

    generate
        if (FORCE_BRAM) begin : gen_bram_w
            (* ram_style = "block" *) reg signed [1:0] w_rom_bram [0:TOTAL_WEIGHTS-1];
            initial $readmemh(WEIGHT_FILE, w_rom_bram);
        end else begin : gen_dist_w
            (* ram_style = "distributed" *) reg signed [1:0] w_rom_dist [0:TOTAL_WEIGHTS-1];
            initial $readmemh(WEIGHT_FILE, w_rom_dist);
        end
    endgenerate

    reg [2:0]  state;
    reg [31:0] neuron_idx;     // 0 .. NUM_NEURONS-1
    reg [31:0] input_idx;      // 0 .. LAYER_NEURON_WIDTH
    reg [31:0] w_addr;         // flat index into w_rom (auto-incrementing)
    reg [1:0]  drain_cnt;      // counts 0,1 during S_DRAIN

    // ================================================================
    //  Per-neuron activation threshold — always present
    //  When all zeros: skip_fc_tap is always 0 → pure TTQ behavior
    // ================================================================
    (* ram_style = "distributed" *)
    reg signed [31:0] act_thresh_rom [0 : NUM_NEURONS - 1];
    initial $readmemh(ACT_THRESH_FILE, act_thresh_rom);
    wire signed [31:0] cur_thresh = act_thresh_rom[neuron_idx];

    // ================================================================
    //  FSM — 3 bits, 8 states (0-7)
    // ================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_FILL  = 3'd1;   // Pipeline priming (1 cycle)
    localparam S_MAC   = 3'd2;   // Ternary accumulate (split pos/neg)
    localparam S_DRAIN = 3'd3;   // Drain 1-stage pipeline (2 cycles)
    localparam S_SCALE = 3'd4;   // NEW: Wp/Wn multiply + bias add
    localparam S_BN    = 3'd5;   // BatchNorm multiply (HAS_BN=1 only)
    localparam S_STORE = 3'd6;   // BN shift + ReLU + store
    localparam S_DONE  = 3'd7;

    // ================================================================
    //  Datapath — 1-stage pipeline (ternary, no multiplier in MAC)
    //  Combinational array read → register → ternary accumulate
    // ================================================================
    reg signed [1:0]           p1_code;   // 2-bit ternary weight code
    reg signed [LAYER_BITS:0]  p1_data;   // registered input data

    // Combinational weight read from whichever ROM exists
    wire signed [1:0] w_rom_combo;
    generate
        if (FORCE_BRAM) begin : gen_bram_rd
            assign w_rom_combo = gen_bram_w.w_rom_bram[w_addr];
        end else begin : gen_dist_rd
            assign w_rom_combo = gen_dist_w.w_rom_dist[w_addr];
        end
    endgenerate

    // Combinational lookahead weight reads
    wire signed [1:0] w_rom_combo_p1;
    wire signed [1:0] w_rom_combo_p2;
    wire signed [1:0] w_rom_combo_p3;
    generate
        if (FORCE_BRAM) begin : gen_bram_la
            assign w_rom_combo_p1 = (w_addr + 1 < TOTAL_WEIGHTS) ? gen_bram_w.w_rom_bram[w_addr + 1] : 2'b00;
            assign w_rom_combo_p2 = (w_addr + 2 < TOTAL_WEIGHTS) ? gen_bram_w.w_rom_bram[w_addr + 2] : 2'b00;
            assign w_rom_combo_p3 = (w_addr + 3 < TOTAL_WEIGHTS) ? gen_bram_w.w_rom_bram[w_addr + 3] : 2'b00;
        end else begin : gen_dist_la
            assign w_rom_combo_p1 = (w_addr + 1 < TOTAL_WEIGHTS) ? gen_dist_w.w_rom_dist[w_addr + 1] : 2'b00;
            assign w_rom_combo_p2 = (w_addr + 2 < TOTAL_WEIGHTS) ? gen_dist_w.w_rom_dist[w_addr + 2] : 2'b00;
            assign w_rom_combo_p3 = (w_addr + 3 < TOTAL_WEIGHTS) ? gen_dist_w.w_rom_dist[w_addr + 3] : 2'b00;
        end
    endgenerate

    // ================================================================
    //  Registered threshold pre-computation (DAAP)
    // ================================================================
    wire signed [LAYER_BITS:0] precomp_act = (input_idx + 1 <= LAYER_NEURON_WIDTH) ? data_in[input_idx + 1] : {(LAYER_BITS+1){1'b0}};
    wire signed [LAYER_BITS:0] abs_precomp_act = (precomp_act < 0) ? -precomp_act : precomp_act;
    wire precomp_below = ($unsigned(abs_precomp_act) < $unsigned(cur_thresh));

    reg cur_below_thresh_r;
    reg skipped_multi_last;

    always @(posedge clk) begin
        if (!rstn || state == S_IDLE || state == S_STORE)
            cur_below_thresh_r <= 1'b0;
        else if (state == S_FILL || state == S_MAC)
            cur_below_thresh_r <= precomp_below;
    end

    // ================================================================
    //  Skip logic — DAAP activation threshold + weight zero check
    //  Skip when weight is 0 or activation is below threshold
    // ================================================================
    wire cur_skip = (w_rom_combo == 2'b00) || (cur_below_thresh_r && !skipped_multi_last);

    // Lookahead checks for weights
    wire skip_p1 = (input_idx + 1 <= LAYER_NEURON_WIDTH) ? (w_rom_combo_p1 == 2'b00) : 1'b0;
    wire skip_p2 = (input_idx + 2 <= LAYER_NEURON_WIDTH) ? (w_rom_combo_p2 == 2'b00) : 1'b0;
    wire skip_p3 = (input_idx + 3 <= LAYER_NEURON_WIDTH) ? (w_rom_combo_p3 == 2'b00) : 1'b0;

    // Compute advance amount (1 to 4)
    wire [2:0] skip_advance;
    assign skip_advance = (!cur_skip) ? 3'd1 :
                          (!skip_p1) ? 3'd1 :
                          (!skip_p2) ? 3'd2 :
                          (!skip_p3) ? 3'd3 : 3'd4;

    always @(posedge clk) begin
        if (!rstn) skipped_multi_last <= 1'b0;
        else       skipped_multi_last <= (state == S_MAC && skip_advance > 3'd1);
    end

    wire [31:0] next_input_idx = input_idx + {29'd0, skip_advance};
    wire advance_past_end = (next_input_idx > LAYER_NEURON_WIDTH);

    // Pipeline validity & Gating — 1-stage
    wire feeding;
    assign feeding = (state == S_FILL) || (state == S_MAC && !cur_skip);

    // Single register stage — clock in weight code and input data when feeding is active
    always @(posedge clk) begin
        if (feeding) begin
            p1_code <= w_rom_combo;
            p1_data <= data_in[input_idx];
        end
    end

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn)
            pipe_s1_valid <= 1'b0;
        else
            pipe_s1_valid <= feeding;
    end

    // ================================================================
    //  Split accumulators — TTQ change
    //   pos_acc: activations at +1 positions (code == 2'b01)
    //   neg_acc: activations at −1 positions (code == 2'b11)
    // ================================================================
    reg signed [LAYER_BITS+24:0] pos_acc;
    reg signed [LAYER_BITS+24:0] neg_acc;

    // Q16.16 extraction before Wp/Wn multiply.
    // The accumulators store sums in Q16.16 format (decimal at bit 16).
    // We take [31:0] to preserve the full Q16.16 value.
    wire signed [31:0] pos_acc_q16;
    wire signed [31:0] neg_acc_q16;
    assign pos_acc_q16 = pos_acc[31:0];
    assign neg_acc_q16 = neg_acc[31:0];

    // ================================================================
    //  BN registers (unchanged from baseline)
    // ================================================================
    reg signed [LAYER_BITS+24:0] biased_reg;
    reg signed [LAYER_BITS+24:0] bn_product_reg;

    // Q16.16 extraction of biased_reg for BN multiply.
    // biased_reg is already in Q16.16 after S_SCALE (>>> 16 was applied).
    wire signed [31:0] biased_q16;
    assign biased_q16 = biased_reg[31:0];

    // Final result: BN output (with BN) or raw biased (without BN)
    wire signed [LAYER_BITS+24:0] final_result;
    assign final_result = HAS_BN ?
        (bn_product_reg + $signed(bn_shift[neuron_idx])) :
        biased_reg;

    // ================================================================
    //  State machine
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state              <= S_IDLE;
            neuron_idx         <= 0;
            input_idx          <= 0;
            w_addr             <= 0;
            pos_acc            <= 0;
            neg_acc            <= 0;
            biased_reg         <= 0;
            bn_product_reg     <= 0;
            drain_cnt          <= 0;
            counter_donestatus <= 0;
        end else begin
            counter_donestatus <= 0;

            case (state)

                // ---- Start first neuron ----
                S_IDLE: begin
                    neuron_idx <= 0;
                    input_idx  <= 0;
                    w_addr     <= 0;
                    pos_acc    <= 0;
                    neg_acc    <= 0;
                    state      <= S_FILL;
                end

                // ---- Pipeline priming: address 0 issued this cycle,
                //      weight/data will be valid next cycle. ----
                S_FILL: begin
                    input_idx <= input_idx + 1;
                    w_addr    <= w_addr + 1;
                    state     <= S_MAC;
                end

                // ---- Ternary accumulate — split into pos/neg ----
                //      Accumulate pipeline output, issue next address.
                //      Transition to drain when all addresses presented.
                S_MAC: begin
                    if (pipe_s1_valid) begin
                        case (p1_code)
                            2'b01:   pos_acc <= pos_acc + p1_data;
                            2'b11:   neg_acc <= neg_acc + p1_data;
                            default: ;
                        endcase
                    end

                    if (advance_past_end || input_idx >= LAYER_NEURON_WIDTH) begin
                        w_addr    <= w_addr + {29'd0, skip_advance};
                        drain_cnt <= 0;
                        state     <= S_DRAIN;
                    end else begin
                        input_idx <= next_input_idx;
                        w_addr    <= w_addr + {29'd0, skip_advance};
                    end
                end

                // ---- Drain: flush the 1-stage pipeline, go to S_SCALE ----
                //  drain_cnt==0: collect last p1 tap into pos/neg acc
                //  drain_cnt==1: transition to S_SCALE
                S_DRAIN: begin
                    drain_cnt <= drain_cnt + 1;

                    if (drain_cnt == 2'd0) begin
                        if (pipe_s1_valid) begin
                            case (p1_code)
                                2'b01:   pos_acc <= pos_acc + p1_data;
                                2'b11:   neg_acc <= neg_acc + p1_data;
                                default: ;
                            endcase
                        end
                    end

                    if (drain_cnt == 2'd1) begin
                        state <= S_SCALE;
                    end
                end

                // ---- S_SCALE — TTQ Wp/Wn multiply (NEW STATE) ----
                //
                //  biased_reg = Wp*pos_acc_q16 - Wn*neg_acc_q16 + b[neuron_idx]
                //
                //  Wp and Wn are scalar (same for all neurons in this layer).
                //  pos_acc_q16 and neg_acc_q16 are per-neuron Q16.16 slices.
                //  Vivado infers 2 DSP48s here, fired once per neuron.
                //  After this state, biased_reg feeds into S_BN exactly as
                //  it did before (biased_q16 truncation is still applied).
                S_SCALE: begin
                    biased_reg <= (($signed(wp) * pos_acc_q16) >>> 16)
                                - (($signed(wn) * neg_acc_q16) >>> 16)
                                + $signed(b[neuron_idx]);
                    if (HAS_BN)
                        state <= S_BN;
                    else
                        state <= S_STORE;
                end

                // ---- BN multiply (HAS_BN=1 only) — UNCHANGED ----
                S_BN: begin
                    bn_product_reg <= ($signed(bn_scale[neuron_idx]) * biased_q16) >>> 16;
                    state          <= S_STORE;
                end

                // ---- Finalise + optional ReLU, store result ----
                S_STORE: begin
                    if (activation_function && final_result <= 0)
                        data_out[neuron_idx] <= {(LAYER_BITS+9){1'b0}};
                    else
                        data_out[neuron_idx] <= final_result[LAYER_BITS+8:0];

                    // Reset split accumulators for next neuron
                    pos_acc    <= 0;
                    neg_acc    <= 0;
                    input_idx  <= 0;

                    if (neuron_idx == NUM_NEURONS - 1)
                        state <= S_DONE;
                    else begin
                        neuron_idx <= neuron_idx + 1;
                        state      <= S_FILL;
                    end
                end

                S_DONE: begin
                    counter_donestatus <= 1;
                end

            endcase
        end
    end

endmodule
