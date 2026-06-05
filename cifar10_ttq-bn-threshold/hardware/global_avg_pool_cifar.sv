`timescale 1ns / 1ps
//============================================================================
// Global Average Pooling (GAP) — CIFAR-10 (BRAM Interface)
//
// Reduces spatial dimensions to 1×1 per channel by averaging all pixels.
//   Input:  IN_CH × IN_H × IN_W values via BRAM address/data interface
//   Output: IN_CH values (Q16.16) — kept as reg array (only 64 entries)
//
// BRAM read latency handling:
//   fm_rd_addr is driven combinationally. fm_rd_data arrives 1 cycle later.
//   After all spatial addresses have been presented for a channel, an
//   extra S_ACC_FLUSH cycle accumulates the final value.
//
// Fixed-point: Q16.16
//============================================================================
(* KEEP_HIERARCHY = "yes" *) module global_avg_pool_cifar #(
    parameter IN_CH     = 64,
    parameter IN_H      = 8,
    parameter IN_W      = 8,
    parameter GAP_SHIFT = 6,       // log2(IN_H × IN_W) = log2(64)
    parameter BITS      = 31
)(
    input  wire                     clk,
    input  wire                     rstn,

    // BRAM read port — reads from conv4's output feature map
    output wire [31:0]              fm_rd_addr,
    input  wire signed [BITS:0]     fm_rd_data,

    // Output — small array, fine as flip-flops (64 entries)
    output reg  signed [BITS:0]     data_out [0 : IN_CH - 1],
    output reg                      done
);

    localparam SPATIAL_SIZE = IN_H * IN_W;   // 64 for 8×8

    // ================================================================
    //  FSM — 5 states
    // ================================================================
    localparam S_IDLE      = 3'd0;
    localparam S_ACC       = 3'd1;
    localparam S_ACC_FLUSH = 3'd2;   // Extra cycle for last BRAM read
    localparam S_STORE     = 3'd3;
    localparam S_DONE      = 3'd4;

    reg [2:0]  state;
    reg [31:0] ch_idx;         // 0..IN_CH-1
    reg [31:0] spatial_idx;    // 0..SPATIAL_SIZE-1

    // Accumulator
    reg signed [BITS+24:0] acc;

    // Pipeline valid — tracks when fm_rd_data contains valid data
    reg acc_pipe_valid;

    // ================================================================
    //  BRAM read address — driven combinationally
    // ================================================================
    assign fm_rd_addr = ch_idx * SPATIAL_SIZE + spatial_idx;

    // Shifted result
    wire signed [BITS+24:0] gap_shifted;
    assign gap_shifted = acc >>> GAP_SHIFT;

    // ================================================================
    //  State machine
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state          <= S_IDLE;
            ch_idx         <= 0;
            spatial_idx    <= 0;
            acc            <= 0;
            done           <= 0;
            acc_pipe_valid <= 0;
        end else begin
            done <= 0;

            // Accumulate whenever we have valid BRAM data
            // (1 cycle after address was presented during S_ACC)
            if (acc_pipe_valid)
                acc <= acc + $signed(fm_rd_data);

            case (state)

                // ---- Initialise ----
                S_IDLE: begin
                    ch_idx         <= 0;
                    spatial_idx    <= 0;
                    acc            <= 0;
                    acc_pipe_valid <= 0;
                    state          <= S_ACC;
                end

                // ---- Present addresses for all spatial positions ----
                S_ACC: begin
                    acc_pipe_valid <= 1;  // fm_rd_data will be valid next cycle

                    if (spatial_idx == SPATIAL_SIZE - 1) begin
                        spatial_idx    <= 0;
                        acc_pipe_valid <= 1;  // One more valid data coming
                        state          <= S_ACC_FLUSH;
                    end else
                        spatial_idx <= spatial_idx + 1;
                end

                // ---- Flush: accumulate the last BRAM read ----
                S_ACC_FLUSH: begin
                    acc_pipe_valid <= 0;  // No more valid data after this
                    state          <= S_STORE;
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
