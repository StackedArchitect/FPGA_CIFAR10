`timescale 1ns / 1ps
//============================================================================
// Sequential FC Layer with BRAM Weight Storage
//
// Processes neurons ONE at a time (serial MAC), storing weights in a single
// 1D BRAM-friendly ROM.  Replaces the parallel `layer` module for synthesis
// on resource-limited FPGAs (e.g., xc7z020).
//
// Savings vs parallel `layer` (FC1 example, 32 neurons × 240 inputs):
//   • Multipliers:  32 → 1  (saves 31 DSP48)
//   • Weight MUXes: 32 × (240:1 × 32b) → eliminated (single BRAM read)
//   • Weight LUTs:  ~4000+ LUT6 → ~7 BRAM36k blocks
//
// 1-stage pipeline (matches TTQ architecture for comparative analysis):
//   Stage 1 (posedge): BRAM read + data MUX → registered cur_weight/cur_data
//   Accumulate (posedge): Q16.16 multiply (combinational) + acc
//
//   S_FILL (1 cyc) → S_MAC (W cyc) → S_DRAIN (2 cyc) → S_STORE (1 cyc)
//   then next neuron or S_DONE.
//
// Cycle count per neuron: LAYER_NEURON_WIDTH + 4
//   FC1: (239 + 4) × 32 = 7,776 cycles
//   FC2: ( 71 + 4) × 10 =   750 cycles
//
// Weight ROM layout (1D, row-major — same order as $readmemh on 2D array):
//   [ neuron_0 weight_0, ..., neuron_0 weight_W,
//     neuron_1 weight_0, ..., neuron_1 weight_W, ... ]
//
// Fixed-point: Q16.16
//============================================================================
module layer_seq_pl #(
    parameter NUM_NEURONS        = 32,
    parameter LAYER_NEURON_WIDTH = 239,   // Number of inputs − 1 (0-indexed)
    parameter LAYER_BITS         = 31,    // Input data bit width
    parameter B_BITS             = 31,    // Bias bit width
    parameter WEIGHT_FILE        = ""     // Path to .mem file for $readmemh
)(
    input  wire                           clk,
    input  wire                           rstn,
    input  wire                           activation_function,  // 1 = ReLU, 0 = none

    input  wire signed [B_BITS:0]         b        [0:NUM_NEURONS-1],
    input  wire signed [LAYER_BITS:0]     data_in  [0:LAYER_NEURON_WIDTH],

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
    //  FSM
    // ================================================================
    localparam S_IDLE  = 3'd0;
    localparam S_FILL  = 3'd1;   // Pipeline priming (1 cycle)
    localparam S_MAC   = 3'd2;   // Multiply-accumulate
    localparam S_DRAIN = 3'd3;   // Drain 1-stage pipeline (2 cycles)
    localparam S_STORE = 3'd4;   // Bias + ReLU + store
    localparam S_DONE  = 3'd5;

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

    // Pipeline validity — 1-stage (matches TTQ architecture)
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

    // Bias addition
    wire signed [LAYER_BITS+24:0] biased;
    assign biased = acc + b[neuron_idx];

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
                        // Last address just captured by stage 1;
                        // advance w_addr past this neuron's weights.
                        w_addr    <= w_addr + 1;
                        drain_cnt <= 0;
                        state     <= S_DRAIN;
                    end else begin
                        input_idx <= input_idx + 1;
                        w_addr    <= w_addr + 1;
                    end
                end

                // ---- Drain: flush the 1-stage pipeline.
                //      2 cycles to collect the last product. ----
                S_DRAIN: begin
                    if (pipe_s1_valid)
                        acc <= acc + p1_product;

                    drain_cnt <= drain_cnt + 1;
                    if (drain_cnt == 2'd1)
                        state <= S_STORE;
                end

                // ---- Bias + optional ReLU, store result ----
                S_STORE: begin
                    if (activation_function && biased <= 0)
                        data_out[neuron_idx] <= {(LAYER_BITS+9){1'b0}};
                    else
                        data_out[neuron_idx] <= biased[LAYER_BITS+8:0];

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
