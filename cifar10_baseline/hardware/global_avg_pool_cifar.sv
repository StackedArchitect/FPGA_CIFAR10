`timescale 1ns / 1ps
//============================================================================
// Global Average Pooling (GAP) — CIFAR-10 Baseline
//
// Reduces spatial dimensions to 1×1 per channel by averaging all pixels.
//   Input:  IN_CH × IN_H × IN_W values (Q16.16)
//   Output: IN_CH values (Q16.16)
//
// For CIFAR-10 baseline:
//   Input  = 64 × 8 × 8 = 4096 values
//   Output = 64 values
//   GAP_SHIFT = log2(8×8) = 6 (divide by 64)
//
// No multipliers needed — just addition and arithmetic right-shift.
// Cycle count: IN_CH × SPATIAL_SIZE + IN_CH + overhead
//   = 64 × 64 + 64 ≈ 4,160 cycles
//
// Fixed-point: Q16.16
//============================================================================
module global_avg_pool_cifar #(
    parameter IN_CH     = 64,
    parameter IN_H      = 8,
    parameter IN_W      = 8,
    parameter GAP_SHIFT = 6,       // log2(IN_H × IN_W) = log2(64)
    parameter BITS      = 31
)(
    input  wire                     clk,
    input  wire                     rstn,

    input  wire signed [BITS:0]     data_in  [0 : IN_CH * IN_H * IN_W - 1],
    output reg  signed [BITS:0]     data_out [0 : IN_CH - 1],
    output reg                      done
);

    localparam SPATIAL_SIZE = IN_H * IN_W;   // 64 for 8×8

    // ================================================================
    //  FSM — 3 states
    // ================================================================
    localparam S_IDLE  = 2'd0;
    localparam S_ACC   = 2'd1;
    localparam S_STORE = 2'd2;
    localparam S_DONE  = 2'd3;

    reg [1:0]  state;
    reg [31:0] ch_idx;         // 0..IN_CH-1
    reg [31:0] spatial_idx;    // 0..SPATIAL_SIZE-1

    // Accumulator — need log2(SPATIAL_SIZE) extra bits above Q16.16
    // For 8×8: 6 extra bits → 38-bit accumulator is sufficient.
    // Using BITS+24 for generous headroom (same as conv accumulators).
    reg signed [BITS+24:0] acc;

    // Data read address
    wire [31:0] read_addr;
    assign read_addr = ch_idx * SPATIAL_SIZE + spatial_idx;

    // Shifted result
    wire signed [BITS+24:0] gap_shifted;
    assign gap_shifted = acc >>> GAP_SHIFT;

    // ================================================================
    //  State machine
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state       <= S_IDLE;
            ch_idx      <= 0;
            spatial_idx <= 0;
            acc         <= 0;
            done        <= 0;
        end else begin
            done <= 0;

            case (state)

                // ---- Initialise ----
                S_IDLE: begin
                    ch_idx      <= 0;
                    spatial_idx <= 0;
                    acc         <= 0;
                    state       <= S_ACC;
                end

                // ---- Accumulate all spatial positions for one channel ----
                S_ACC: begin
                    acc <= acc + $signed(data_in[read_addr]);

                    if (spatial_idx == SPATIAL_SIZE - 1) begin
                        spatial_idx <= 0;
                        state       <= S_STORE;
                    end else
                        spatial_idx <= spatial_idx + 1;
                end

                // ---- Divide by SPATIAL_SIZE (right-shift) and store ----
                S_STORE: begin
                    data_out[ch_idx] <= gap_shifted[BITS:0];
                    acc              <= 0;

                    if (ch_idx == IN_CH - 1)
                        state <= S_DONE;
                    else begin
                        ch_idx <= ch_idx + 1;
                        state  <= S_ACC;
                    end
                end

                // ---- Done — pulse done signal ----
                S_DONE: begin
                    done <= 1;
                end

            endcase
        end
    end

endmodule
