`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer — Full-Precision + BatchNorm (CIFAR-10 Baseline)
//
// Adapted from cnn2d_baseline/hardware_sequential_bn/layer_seq_bn.sv
// with module rename. Logic is identical.
//
// Features:
//   • HAS_BN parameter: 1 for FC1 (BN after MAC), 0 for FC2 (no BN)
//   • Sequential MAC with 1 multiplier per neuron (time-shared)
//   • Weight ROM via $readmemh (BRAM-inferred)
//   • PAD=20 zero-padding convention
//
// For CIFAR-10 baseline:
//   FC1: NUM_NEURONS=256, LAYER_NEURON_WIDTH=103 (PAD+64+PAD-1)
//        HAS_BN=1, cycles per neuron = 103 + 5 = 108
//        Total: 108 × 256 = 27,648 cycles
//   FC2: NUM_NEURONS=10,  LAYER_NEURON_WIDTH=295 (PAD+256+PAD-1)
//        HAS_BN=0, cycles per neuron = 295 + 4 = 299
//        Total: 299 × 10 = 2,990 cycles
//
// Fixed-point: Q16.16
//============================================================================
module layer_seq_cifar #(
    parameter NUM_NEURONS        = 256,
    parameter LAYER_NEURON_WIDTH = 103,    // Number of inputs − 1 (0-indexed)
    parameter LAYER_BITS         = 31,     // Input data bit width
    parameter B_BITS             = 31,     // Bias bit width
    parameter HAS_BN             = 1,      // 1 = FC1 (BN after MAC), 0 = FC2
    parameter WEIGHT_FILE        = ""      // Path to .mem file for $readmemh
)(
    input  wire                           clk,
    input  wire                           rstn,
    input  wire                           activation_function,  // 1 = ReLU, 0 = none

    input  wire signed [B_BITS:0]         b        [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]     data_in  [0:LAYER_NEURON_WIDTH],

    // Folded BN parameters (Q16.16 per neuron, ignored when HAS_BN=0)
    input  wire signed [31:0]             bn_scale [0:NUM_NEURONS-1],
    input  wire signed [31:0]             bn_shift [0:NUM_NEURONS-1],

    output reg  signed [LAYER_BITS+8:0]   data_out [0:NUM_NEURONS-1],
    output reg                            counter_donestatus
);

    // ================================================================
    //  Weight ROM — 1D flat array, BRAM-inferred
    // ================================================================
    localparam NUM_INPUTS    = LAYER_NEURON_WIDTH + 1;
    localparam TOTAL_WEIGHTS = NUM_NEURONS * NUM_INPUTS;

    (* ram_style = "block" *) reg signed [31:0] w_rom [0:TOTAL_WEIGHTS-1];
    initial $readmemh(WEIGHT_FILE, w_rom);

    // ================================================================
    //  FSM — 3 bits, 7 states (0-6)
    // ================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_FILL  = 3'd1;   // Pipeline priming (1 cycle)
    localparam S_MAC   = 3'd2;   // Multiply-accumulate
    localparam S_DRAIN = 3'd3;   // Drain 1-stage pipeline (2 cycles)
    localparam S_BN    = 3'd4;   // BatchNorm multiply (HAS_BN=1 only)
    localparam S_STORE = 3'd5;   // BN shift + ReLU + store
    localparam S_DONE  = 3'd6;

    reg [2:0]  state;
    reg [31:0] neuron_idx;     // 0 .. NUM_NEURONS-1
    reg [31:0] input_idx;      // 0 .. LAYER_NEURON_WIDTH
    reg [31:0] w_addr;         // flat index into w_rom (auto-incrementing)
    reg [1:0]  drain_cnt;      // counts 0,1 during S_DRAIN

    // ================================================================
    //  Datapath — registered BRAM read + registered data MUX
    //  Q16.16 multiply is combinational (1-stage pipeline).
    // ================================================================
    reg signed [31:0]          cur_weight;
    reg signed [LAYER_BITS:0]  cur_data;

    always @(posedge clk) begin
        cur_weight <= w_rom[w_addr];
        cur_data   <= data_in[input_idx];
    end

    // Combinational Q16.16 multiply (accumulated via pipe_s1_valid)
    wire signed [LAYER_BITS+32:0] full_product;
    assign full_product = cur_weight * cur_data;

    wire signed [LAYER_BITS+16:0] p1_product;
    assign p1_product = full_product >>> 16;

    // Pipeline validity — 1-stage (matches baseline architecture)
    wire feeding;
    assign feeding = (state == S_FILL) || (state == S_MAC);

    reg pipe_s1_valid;
    always @(posedge clk) begin
        if (!rstn)
            pipe_s1_valid <= 1'b0;
        else
            pipe_s1_valid <= feeding;
    end

    // Accumulator
    reg signed [LAYER_BITS+24:0] acc;

    // ================================================================
    //  BN registers
    // ================================================================
    reg signed [LAYER_BITS+24:0] biased_reg;
    reg signed [LAYER_BITS+24:0] bn_product_reg;

    // Q16.16 extraction of biased_reg for BN multiply.
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
            acc                <= 0;
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
                    acc        <= 0;
                    state      <= S_FILL;
                end

                // ---- Pipeline priming: address 0 issued this cycle,
                //      weight/data will be valid next cycle. ----
                S_FILL: begin
                    input_idx <= input_idx + 1;
                    w_addr    <= w_addr + 1;
                    state     <= S_MAC;
                end

                // ---- Accumulate pipeline output (p1_product), issue
                //      next address.  Transition to drain when all
                //      addresses have been presented. ----
                S_MAC: begin
                    if (pipe_s1_valid)
                        acc <= acc + p1_product;

                    if (input_idx == LAYER_NEURON_WIDTH) begin
                        w_addr    <= w_addr + 1;
                        drain_cnt <= 0;
                        state     <= S_DRAIN;
                    end else begin
                        input_idx <= input_idx + 1;
                        w_addr    <= w_addr + 1;
                    end
                end

                // ---- Drain: flush the 1-stage pipeline.
                //      2 cycles to collect the last product.
                //      At drain_cnt==1: register biased value,
                //      transition to S_BN or S_STORE. ----
                S_DRAIN: begin
                    if (pipe_s1_valid)
                        acc <= acc + p1_product;

                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1) begin
                        biased_reg <= acc + $signed(b[neuron_idx]);
                        if (HAS_BN)
                            state <= S_BN;
                        else
                            state <= S_STORE;
                    end
                end

                // ---- BN multiply (HAS_BN=1 only) ----
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

                    acc       <= 0;
                    input_idx <= 0;

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
