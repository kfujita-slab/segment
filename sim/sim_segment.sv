`timescale 1ns/1ps
`default_nettype none

module sim_likelihood();
localparam integer SIM_CYCLES = 2000;
localparam real    CLOCK_PERIOD = 10; // 20ns(50MHz)

//-------------------------------------------------------------------------
// Parameters
//------------------------------------------------------------------------- 
// localparam int     H_DISP     = 720;
// localparam int     H_FP       = 16;
// localparam int     H_SYNC     = 62;
// localparam int     H_BP       = 60;
// localparam int     H_BLANK    = H_FP + H_SYNC + H_BP;
// localparam int     H_TOTAL    = H_DISP + H_BLANK;

// localparam int     V_DISP     = 480;
// localparam int     V_FP       = 9;
// localparam int     V_SYNC     = 6;
// localparam int     V_BP       = 30;
// localparam int     V_BLANK    = V_FP + V_SYNC + V_BP;
// localparam int     V_TOTAL    = V_DISP + V_BLANK;

// localparam int     WIDTH      = H_TOTAL;
// localparam int     HEIGHT     = V_TOTAL;
// localparam int     W_WIDTH    = 640;
// localparam int     W_HEIGHT   = V_DISP;

// localparam int     TARGET_HUE = 40;
// localparam int     R_MIN      = 64;


//localparam int     H_DISP          = 720;
localparam int     H_DISP = 640;
localparam int     H_FP = 0;
localparam int     H_SYNC = 0;
localparam int     H_BP = 0;
localparam int     H_BLANK = H_FP + H_SYNC + H_BP;
localparam int     H_TOTAL = H_DISP + H_BLANK;

//localparam int     V_DISP        = 480;
localparam int     V_DISP = 480;
localparam int     V_FP = 0;
localparam int     V_SYNC = 0;
localparam int     V_BP = 0;
localparam int     V_BLANK = V_FP + V_SYNC + V_BP;
localparam int     V_TOTAL = V_DISP + V_BLANK;

localparam int     WIDTH = H_TOTAL;
localparam int     HEIGHT = V_TOTAL;

localparam int     TARGET_HUE = 40;
localparam int     R_MIN = 64;

localparam int     SEED = 12345678;
localparam int     SEED_DIFF = 12345*3;
localparam int     RP_NUM = 5;
localparam int     VP_NUM = 3;

localparam int     MAX_COORD_VAL = 30;
localparam int     MIN_COORD_VAL = -30;
localparam int     MAX_VEL_VAL = 10;
localparam int     MIN_VEL_VAL = -10;





logic 		    clk;
logic 		    n_rst;

logic [23:0] 		    in_y;
//logic [7:0] 		    g_data_in;
//logic [7:0] 		    b_data_in;
logic [$clog2(WIDTH+1)-1:0] hcount_in;
logic [$clog2(HEIGHT+1)-1:0] vcount_in;

wire [1:0] 		       out_y;
wire [$clog2(WIDTH+1)-1:0]  hcount_out;
wire [$clog2(HEIGHT+1)-1:0] vcount_out;



top segment_inst
(
    .clock(clk),
    .n_rst(n_rst),
    .in_y(in_y),
    .in_hcnt(hcount_in),
    .in_vcnt(vcount_in),
    .out_y(out_y),
    .out_hcnt(hcount_out),
    .out_vcnt(vcount_out)
);

initial begin
    clk <= 1'b0;
    repeat (SIM_CYCLES) begin
        #(CLOCK_PERIOD / 2)
        clk <= 1'b1;
        #(CLOCK_PERIOD / 2)
        clk <= 1'b0;
    end
    $finish;
end

initial begin
    n_rst <= 1'b0;
    #(CLOCK_PERIOD)
    n_rst <= 1'b1;
end

initial begin
    #(CLOCK_PERIOD * 3 / 2)
    hcount_in <= '0;
    forever begin
        #(CLOCK_PERIOD)
        if (hcount_in == H_TOTAL - 1)           
            hcount_in <= '0;
        else
            hcount_in <= hcount_in + 1'b1;         
    end
end

initial begin
    #(CLOCK_PERIOD *3 / 2)
    vcount_in <= '0;
    forever begin
        #(CLOCK_PERIOD)
        if (vcount_in == H_TOTAL - 1) begin
            if (vcount_in == V_TOTAL - 1)
                vcount_in <= '0;
            else
                vcount_in <= vcount_in + 1'b1;
        end
    end
end

reg[7:0] i;

initial begin
    i<='0;
    #(CLOCK_PERIOD/2)
    repeat(SIM_CYCLES) begin
        #(CLOCK_PERIOD)
        in_y <= $urand_range(16777215,0);
        print();
    end
end

initial begin
    $shm_open("simulation/likelihood.shm");
    $shm_probe(likelihood_inst, "A");
end

task print();
    $write("out_y = %d",out_y);
endtask

endmodule // sim_likelihood

`default_nettype wire
