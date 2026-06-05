`timescale 1ns / 1ps
//============================================================================
// Activation Mask Generator — Threshold-based 1-Pass Gating Mask
//
// Sequentially reads the feature map from BRAM (1-cycle read latency),
// decodes the channel index combinationally, compares the absolute value
// to the per-channel activation threshold, and writes a 1-bit status
// directly to a flat output register vector.
//
// Latency: N + 2 clock cycles
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================
(* KEEP_HIERARCHY = "yes" *) module act_mask_gen_thresh #(
    parameter IN_H        = 16,
    parameter IN_W        = 16,
    parameter IN_CH       = 32,
    parameter BITS        = 31,
    parameter LOG2_HW     = 8,   // Log2 of (IN_H * IN_W)
    parameter THRESH_FILE = "",  // Path to per-channel threshold ROM
    // Derived — do not override
    parameter MASK_ADDR_W = $clog2(IN_CH * IN_H * IN_W) > 0 ? $clog2(IN_CH * IN_H * IN_W) : 1
)(
    input  wire                               clk,
    input  wire                               rstn,
    input  wire                               start,
 
    // BRAM read interface
    output reg  [31:0]                        bram_rd_addr,
    input  wire signed [BITS:0]               bram_rd_data,
 
    // Mask write port — consumer stores these in its own LUTRAM
    output wire                               mask_wr_en_o,
    output wire [MASK_ADDR_W-1:0]             mask_wr_addr_o,
    output wire                               mask_wr_data_o,
    output reg                                done
);
 
    localparam TOTAL_SIZE = IN_CH * IN_H * IN_W;
 
    // Per-channel activation threshold ROM (distributed)
    (* ram_style = "distributed" *) reg signed [31:0] thresh_rom [0 : IN_CH - 1];
    initial begin
        if (THRESH_FILE != "") begin
            $readmemh(THRESH_FILE, thresh_rom);
        end else begin
            for (integer c = 0; c < IN_CH; c = c + 1) begin
                thresh_rom[c] = 32'sd0;
            end
        end
    end
 
    // FSM States
    localparam S_IDLE  = 2'd0;
    localparam S_READ  = 2'd1;
    localparam S_PIPE  = 2'd2;
    localparam S_DONE  = 2'd3;
 
    reg [1:0] state;
    reg [31:0] addr_cnt;
 
    // BRAM pipeline delay registers
    reg        p_valid_d1;
    reg        p_valid_d2;
    reg [31:0] p_addr_d1;
    reg [31:0] p_addr_d2;
 
    reg                               mask_wr_en;
    reg [MASK_ADDR_W-1:0]             mask_wr_addr;
    reg                               mask_wr_data;
 
    assign mask_wr_en_o   = mask_wr_en;
    assign mask_wr_addr_o = mask_wr_addr;
    assign mask_wr_data_o = mask_wr_data;
 
    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            bram_rd_addr <= 0;
            done         <= 0;
            addr_cnt     <= 0;
            p_valid_d1   <= 0;
            p_valid_d2   <= 0;
            p_addr_d1    <= 0;
            p_addr_d2    <= 0;
            mask_wr_en   <= 0;
            mask_wr_addr <= 0;
            mask_wr_data <= 0;
        end else begin
            done         <= 0;
            mask_wr_en   <= 0;
            mask_wr_addr <= 0;
            mask_wr_data <= 0;
 
            case (state)
                S_IDLE: begin
                    if (start) begin
                        state      <= S_READ;
                        addr_cnt   <= 0;
                        p_valid_d1 <= 0;
                        p_valid_d2 <= 0;
                    end
                end
 
                // Read BRAM sequentially
                S_READ: begin
                    bram_rd_addr <= addr_cnt;
                    p_addr_d1    <= addr_cnt;
                    p_valid_d1   <= 1'b1;
 
                    p_addr_d2    <= p_addr_d1;
                    p_valid_d2   <= p_valid_d1;
 
                    // Process pipeline output from BRAM (2-cycle latency)
                    if (p_valid_d2) begin
                        automatic logic signed [BITS:0] val = bram_rd_data;
                        automatic logic signed [BITS:0] abs_val = (val < 0) ? -val : val;
                        automatic logic [31:0] channel = p_addr_d2 >> LOG2_HW;
 
                        mask_wr_en   <= 1'b1;
                        mask_wr_addr <= p_addr_d2[MASK_ADDR_W-1:0];
                        if ($unsigned(abs_val) >= $unsigned(thresh_rom[channel][BITS:0])) begin
                            mask_wr_data <= 1'b1;
                        end else begin
                            mask_wr_data <= 1'b0;
                        end
                    end
 
                    if (addr_cnt == TOTAL_SIZE - 1) begin
                        state    <= S_PIPE;
                        addr_cnt <= 0;
                    end else begin
                        addr_cnt <= addr_cnt + 1;
                    end
                end
 
                // Drain last pipeline stages
                S_PIPE: begin
                    p_valid_d1 <= 1'b0;
                    p_addr_d2  <= p_addr_d1;
                    p_valid_d2 <= p_valid_d1;
 
                    if (p_valid_d2) begin
                        automatic logic signed [BITS:0] val = bram_rd_data;
                        automatic logic signed [BITS:0] abs_val = (val < 0) ? -val : val;
                        automatic logic [31:0] channel = p_addr_d2 >> LOG2_HW;
 
                        mask_wr_en   <= 1'b1;
                        mask_wr_addr <= p_addr_d2[MASK_ADDR_W-1:0];
                        if ($unsigned(abs_val) >= $unsigned(thresh_rom[channel][BITS:0])) begin
                            mask_wr_data <= 1'b1;
                        end else begin
                            mask_wr_data <= 1'b0;
                        end
                    end
 
                    if (!p_valid_d1 && !p_valid_d2) begin
                        state <= S_DONE;
                    end
                end
 
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
 
endmodule
