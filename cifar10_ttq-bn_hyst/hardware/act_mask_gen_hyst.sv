`timescale 1ns / 1ps
//============================================================================
// Activation Mask Generator — Spatial Hysteresis (v5 — LUTRAM-ready)
//
// 2-pass spatial hysteresis algorithm:
//   Pass 1 (Classify): Pipelined read of feature-map BRAM, classify into
//     ACTIVE / UNCERTAIN / INACTIVE, write status to Block RAM.
//   Pass 2 (Resolve): Scan status BRAM. For UNCERTAIN pixels, sequentially
//     read <=4 cardinal neighbours (2 cycles each). If >=2 are ACTIVE,
//     resolve as ACTIVE.
//
// v5: Mask output exported via write-only signals (mask_wr_en/addr/data).
//     Consumer (conv module) stores the mask in its own distributed RAM
//     (LUTRAM) for zero-latency combinational reads.
//
// Simulation: add  -L xpm  to xelab command.
// Target: XC7Z020CLG484-1 @ 40 MHz
//============================================================================

(* KEEP_HIERARCHY = "yes" *) module act_mask_gen_hyst #(
    parameter IN_H      = 16,
    parameter IN_W      = 16,
    parameter IN_CH     = 32,
    parameter BITS      = 31,
    parameter LOG2_W    = 4,
    parameter LOG2_HW   = 8,
    // Derived — do not override
    parameter MASK_ADDR_W = $clog2(IN_CH * IN_H * IN_W) > 0
                          ? $clog2(IN_CH * IN_H * IN_W) : 1
)(
    input  wire                               clk,
    input  wire                               rstn,
    input  wire                               start,

    input  wire signed [31:0]                 T_L,
    input  wire signed [31:0]                 T_H,

    output reg  [31:0]                        bram_rd_addr,
    input  wire signed [BITS:0]               bram_rd_data,

    // Mask write port — consumer stores these in its own LUTRAM
    output wire                               mask_wr_en_o,
    output wire [MASK_ADDR_W-1:0]             mask_wr_addr_o,
    output wire                               mask_wr_data_o,

    output reg                                done
);

    localparam TOTAL_SIZE = IN_CH * IN_H * IN_W;
    localparam ADDR_W     = MASK_ADDR_W;
    localparam [ADDR_W-1:0] LAST_ADDR = TOTAL_SIZE - 1;

    // Status codes
    localparam [1:0] INACTIVE  = 2'd0;
    localparam [1:0] UNCERTAIN = 2'd1;
    localparam [1:0] ACTIVE    = 2'd2;

    // FSM states
    localparam [3:0] S_IDLE           = 4'd0;
    localparam [3:0] S_PASS1_READ     = 4'd1;
    localparam [3:0] S_PASS1_PIPE     = 4'd2;
    localparam [3:0] S_PASS2_ADDR     = 4'd3;
    localparam [3:0] S_PASS2_CHECK    = 4'd4;
    localparam [3:0] S_PASS2_NBR_ADDR = 4'd5;
    localparam [3:0] S_PASS2_NBR_READ = 4'd6;
    localparam [3:0] S_DONE           = 4'd7;

    // FSM registers
    reg [3:0]          state;
    reg [ADDR_W-1:0]   addr_cnt;
    reg                p1_valid_d1, p1_valid_d2;
    reg [ADDR_W-1:0]   p1_addr_d1,  p1_addr_d2;
    reg [1:0]          nbr_idx;
    reg [2:0]          active_count;

    // ================================================================
    //  Pass 1 — Classification (combinational)
    // ================================================================
    wire signed [BITS:0] p1_val     = bram_rd_data;
    wire signed [BITS:0] p1_abs_val = (p1_val < 0) ? -p1_val : p1_val;
    wire [1:0] p1_class = ($unsigned(p1_abs_val) > $unsigned(T_H[BITS:0])) ? ACTIVE  :
                           ($unsigned(p1_abs_val) >= $unsigned(T_L[BITS:0])) ? UNCERTAIN :
                           INACTIVE;

    // ================================================================
    //  Status RAM — xpm_memory_sdpram (guaranteed Block RAM)
    //  Port A = write (status classification during Pass 1)
    //  Port B = read  (status lookup during Pass 2)
    //  1-cycle registered read latency, common clock.
    // ================================================================
    wire p1_wr_en = p1_valid_d2 && (state == S_PASS1_READ || state == S_PASS1_PIPE);

    // Coordinate decode
    wire [ADDR_W-1:0] cur_r   = (addr_cnt & ((1 << LOG2_HW) - 1)) >> LOG2_W;
    wire [ADDR_W-1:0] cur_col = addr_cnt & ((1 << LOG2_W) - 1);
    wire has_up    = (cur_r   > 0);
    wire has_down  = (cur_r   < (IN_H - 1));
    wire has_left  = (cur_col > 0);
    wire has_right = (cur_col < (IN_W - 1));

    wire [ADDR_W-1:0] nbr_addr_up    = addr_cnt - (1 << LOG2_W);
    wire [ADDR_W-1:0] nbr_addr_down  = addr_cnt + (1 << LOG2_W);
    wire [ADDR_W-1:0] nbr_addr_left  = addr_cnt - 1;
    wire [ADDR_W-1:0] nbr_addr_right = addr_cnt + 1;

    reg [ADDR_W-1:0] cur_nbr_addr;
    reg              cur_nbr_valid;
    always @(*) begin
        case (nbr_idx)
            2'd0:    begin cur_nbr_addr = nbr_addr_up;    cur_nbr_valid = has_up;    end
            2'd1:    begin cur_nbr_addr = nbr_addr_down;  cur_nbr_valid = has_down;  end
            2'd2:    begin cur_nbr_addr = nbr_addr_left;  cur_nbr_valid = has_left;  end
            default: begin cur_nbr_addr = nbr_addr_right; cur_nbr_valid = has_right; end
        endcase
    end

    // Read-address mux (combinational)
    wire [ADDR_W-1:0] status_rd_addr = (state == S_PASS2_NBR_ADDR && cur_nbr_valid)
                                        ? cur_nbr_addr : addr_cnt;

    // ----------------------------------------------------------------
    //  XPM Simple Dual-Port Block RAM — Status
    //  Port A = write (status classification during Pass 1)
    //  Port B = read  (status lookup during Pass 2)
    //  1-cycle registered read latency, common clock.
    // ----------------------------------------------------------------
    wire [1:0] status_rd_data;

    xpm_memory_sdpram #(
        .ADDR_WIDTH_A            (ADDR_W),
        .ADDR_WIDTH_B            (ADDR_W),
        .AUTO_SLEEP_TIME         (0),
        .BYTE_WRITE_WIDTH_A      (2),
        .CLOCKING_MODE           ("common_clock"),
        .ECC_MODE                ("no_ecc"),
        .MEMORY_INIT_FILE        ("none"),
        .MEMORY_INIT_PARAM       ("0"),
        .MEMORY_OPTIMIZATION     ("true"),
        .MEMORY_PRIMITIVE        ("block"),
        .MEMORY_SIZE             (TOTAL_SIZE * 2),    // bits
        .MESSAGE_CONTROL         (0),
        .READ_DATA_WIDTH_B       (2),
        .READ_LATENCY_B          (1),
        .READ_RESET_VALUE_B      ("0"),
        .RST_MODE_A              ("SYNC"),
        .RST_MODE_B              ("SYNC"),
        .SIM_ASSERT_CHK          (0),
        .USE_EMBEDDED_CONSTRAINT (0),
        .USE_MEM_INIT            (0),
        .USE_MEM_INIT_MMI        (0),
        .WAKEUP_TIME             ("disable_sleep"),
        .WRITE_DATA_WIDTH_A      (2),
        .WRITE_MODE_B            ("no_change")
    ) u_status_bram (
        .dbiterrb       (),
        .doutb          (status_rd_data),
        .sbiterrb       (),
        .addra          (p1_addr_d2),
        .addrb          (status_rd_addr),
        .clka           (clk),
        .clkb           (clk),
        .dina           (p1_class),
        .ena            (p1_wr_en),
        .enb            (1'b1),
        .injectdbiterra (1'b0),
        .injectsbiterra (1'b0),
        .regceb         (1'b1),
        .rstb           (1'b0),
        .sleep          (1'b0),
        .wea            (p1_wr_en)
    );

    // ================================================================
    //  Pass 2 — Neighbour active count (combinational)
    // ================================================================
    wire [2:0] nbr_new_count = (status_rd_data == ACTIVE)
                                ? (active_count + 3'd1) : active_count;

    // ================================================================
    //  Mask resolution signals — exported to consumer via output ports.
    //  Consumer stores these in its own distributed RAM (LUTRAM).
    // ================================================================
    reg  mask_wr_en;
    reg  mask_wr_bit;

    assign mask_wr_en_o   = mask_wr_en;
    assign mask_wr_addr_o = addr_cnt;
    assign mask_wr_data_o = mask_wr_bit;

    always @(*) begin
        mask_wr_en  = 1'b0;
        mask_wr_bit = 1'b0;
        case (state)
            S_PASS2_CHECK: begin
                if (status_rd_data == ACTIVE) begin
                    mask_wr_en  = 1'b1;
                    mask_wr_bit = 1'b1;
                end else if (status_rd_data == INACTIVE) begin
                    mask_wr_en  = 1'b1;
                    mask_wr_bit = 1'b0;
                end
            end
            S_PASS2_NBR_ADDR: begin
                // All neighbours exhausted with this one invalid → resolve
                if (!cur_nbr_valid && nbr_idx == 2'd3) begin
                    mask_wr_en  = 1'b1;
                    mask_wr_bit = (active_count >= 3'd2) ? 1'b1 : 1'b0;
                end
            end
            S_PASS2_NBR_READ: begin
                if (nbr_new_count >= 3'd2) begin
                    mask_wr_en  = 1'b1;
                    mask_wr_bit = 1'b1;
                end else if (nbr_idx == 2'd3) begin
                    mask_wr_en  = 1'b1;
                    mask_wr_bit = 1'b0;
                end
            end
            default: ;
        endcase
    end

    // ================================================================
    //  Main FSM
    // ================================================================
    always @(posedge clk) begin
        if (!rstn) begin
            state        <= S_IDLE;
            bram_rd_addr <= 32'd0;
            done         <= 1'b0;
            addr_cnt     <= {ADDR_W{1'b0}};
            p1_valid_d1  <= 1'b0;
            p1_valid_d2  <= 1'b0;
            p1_addr_d1   <= {ADDR_W{1'b0}};
            p1_addr_d2   <= {ADDR_W{1'b0}};
            nbr_idx      <= 2'd0;
            active_count <= 3'd0;
        end else begin
            done <= 1'b0;

            case (state)

                // ================================================
                //  IDLE
                // ================================================
                S_IDLE: begin
                    if (start) begin
                        state       <= S_PASS1_READ;
                        addr_cnt    <= {ADDR_W{1'b0}};
                        p1_valid_d1 <= 1'b0;
                        p1_valid_d2 <= 1'b0;
                    end
                end

                // ================================================
                //  PASS 1: pipelined feature-map read + classify
                // ================================================
                S_PASS1_READ: begin
                    bram_rd_addr <= addr_cnt;
                    p1_addr_d1   <= addr_cnt;
                    p1_valid_d1  <= 1'b1;

                    p1_addr_d2  <= p1_addr_d1;
                    p1_valid_d2 <= p1_valid_d1;

                    if (addr_cnt == LAST_ADDR) begin
                        state    <= S_PASS1_PIPE;
                        addr_cnt <= {ADDR_W{1'b0}};
                    end else begin
                        addr_cnt <= addr_cnt + 1;
                    end
                end

                // ================================================
                //  Drain pipeline
                // ================================================
                S_PASS1_PIPE: begin
                    p1_valid_d1 <= 1'b0;
                    p1_addr_d2  <= p1_addr_d1;
                    p1_valid_d2 <= p1_valid_d1;

                    if (!p1_valid_d1 && !p1_valid_d2) begin
                        state    <= S_PASS2_ADDR;
                        addr_cnt <= {ADDR_W{1'b0}};
                    end
                end

                // ================================================
                //  PASS 2: present current-pixel address to BRAM
                // ================================================
                S_PASS2_ADDR: begin
                    state <= S_PASS2_CHECK;
                end

                // ================================================
                //  PASS 2: read classification, branch
                // ================================================
                S_PASS2_CHECK: begin
                    case (status_rd_data)
                        ACTIVE, INACTIVE: begin
                            if (addr_cnt == LAST_ADDR)
                                state <= S_DONE;
                            else begin
                                addr_cnt <= addr_cnt + 1;
                                state    <= S_PASS2_ADDR;
                            end
                        end
                        default: begin // UNCERTAIN
                            nbr_idx      <= 2'd0;
                            active_count <= 3'd0;
                            state        <= S_PASS2_NBR_ADDR;
                        end
                    endcase
                end

                // ================================================
                //  PASS 2: present neighbour address or skip
                // ================================================
                S_PASS2_NBR_ADDR: begin
                    if (cur_nbr_valid) begin
                        state <= S_PASS2_NBR_READ;
                    end else begin
                        if (nbr_idx == 2'd3) begin
                            if (addr_cnt == LAST_ADDR)
                                state <= S_DONE;
                            else begin
                                addr_cnt <= addr_cnt + 1;
                                state    <= S_PASS2_ADDR;
                            end
                        end else begin
                            nbr_idx <= nbr_idx + 1;
                        end
                    end
                end

                // ================================================
                //  PASS 2: read neighbour, update count
                // ================================================
                S_PASS2_NBR_READ: begin
                    if (nbr_new_count >= 3'd2) begin
                        active_count <= nbr_new_count;
                        if (addr_cnt == LAST_ADDR)
                            state <= S_DONE;
                        else begin
                            addr_cnt <= addr_cnt + 1;
                            state    <= S_PASS2_ADDR;
                        end
                    end else if (nbr_idx == 2'd3) begin
                        active_count <= nbr_new_count;
                        if (addr_cnt == LAST_ADDR)
                            state <= S_DONE;
                        else begin
                            addr_cnt <= addr_cnt + 1;
                            state    <= S_PASS2_ADDR;
                        end
                    end else begin
                        active_count <= nbr_new_count;
                        nbr_idx      <= nbr_idx + 1;
                        state        <= S_PASS2_NBR_ADDR;
                    end
                end

                // ================================================
                //  DONE
                // ================================================
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

            endcase
        end
    end

endmodule
