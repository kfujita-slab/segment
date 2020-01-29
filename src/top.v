//---------------------------------------------------------
// <top>
//  cnn based segmentation system module
//
//---------------------------------------------------------
// kfujita
// Vertion 1.0 (01_20, 2020)
//---------------------------------------------------------
`default_nettype none
`timescale 1ns/1ns

module top
(
    clock,
    n_rst,
    in_y,
    in_vcnt,
    in_hcnt,
    out_y,
    out_vcnt,
    out_hcnt
);

// parameters
localparam integer WIDTH0    = 480;
localparam integer HEIGHT0   = 640;
localparam integer W_WIDTH0  = 480; // ?
localparam integer W_HEIGHT0 = 640; // ?

localparam integer OUT_BITW  = 13;
localparam integer UNITS     = 12;

// other parameters
localparam integer WIDTH1    = WIDTH0    / 2;
localparam integer HEIGHT1   = HEIGHT0   / 2;
localparam integer W_WIDTH1  = W_WIDTH0  / 2;
localparam integer W_HEIGHT1 = W_HEIGHT0 / 2;

localparam integer WIDTH2    = WIDTH1    / 2;
localparam integer HEIGHT2   = HEIGHT1   / 2;
localparam integer W_WIDTH2  = W_WIDTH1  / 2;
localparam integer W_HEIGHT2 = W_HEIGHT1 / 2;

localparam integer WIDTH3    = WIDTH2    / 2;
localparam integer HEIGHT3   = HEIGHT2   / 2;
localparam integer W_WIDTH3  = W_WIDTH2  / 2;
localparam integer W_HEIGHT3 = W_HEIGHT2 / 2;

localparam integer WIDTH4    = WIDTH3    / 2;
localparam integer HEIGHT4   = HEIGHT3   / 2;
localparam integer W_WIDTH4  = W_WIDTH3  / 2;
localparam integer W_HEIGHT4 = W_HEIGHT3 / 2;

localparam integer H_BITW0   = log2(W_WIDTH0);
localparam integer V_BITW0   = log2(W_HEIGHT0);
localparam integer H_BITW1   = log2(W_WIDTH1);
localparam integer V_BITW1   = log2(W_HEIGHT1);
localparam integer H_BITW2   = log2(W_WIDTH2);
localparam integer V_BITW2   = log2(W_HEIGHT2);
localparam integer H_BITW3   = log2(W_WIDTH3);
localparam integer V_BITW3   = log2(W_HEIGHT3);
localparam integer H_BITW4   = log2(W_WIDTH4);
localparam integer V_BITW4   = log2(W_HEIGHT4);

// inputs/outputs
input wire                   clock, n_rst;
input wire  [8*3-1:0]        in_y;
input wire  [V_BITW0-1:0]    in_vcnt;
input wire  [H_BITW0-1:0]    in_hcnt;
output wire [1:0]            out_y;
output wire [V_BITW0-1:0]    out_vcnt;
output wire [H_BITW0-1:0]    out_hcnt;

// LEVEL0 Left
wire [V_BITW0-1:0]        ext_vcnt;
wire [H_BITW0-1:0]        ext_hcnt;
wire [0:OUT_BITW*UNITS-1] ext_out;
extnet
#(
    .HEIGHT(HEIGHT0),
    .WIDTH(WIDTH0),
    .W_HEIGHT(W_HEIGHT0),
    .W_WIDTH(W_WIDTH0),
    .UINT_BITW(8),
    .PATCH_SIZE(3)
)
extnet_inst
(
    .clock(clock),
    .n_rst(n_rst),
    .in_y(in_y),
    .in_vcnt(in_vcnt),
    .in_hcnt(in_hcnt),
    .out_y(ext_out),
    .out_vcnt(ext_vcnt),
    .out_hcnt(ext_hcnt)
);

wire                      pool_enable1;
wire [V_BITW1-1:0]        pool_vcnt1;
wire [H_BITW1-1:0]        pool_hcnt1;
wire [0:OUT_BITW*UNITS-1] pool_out1;
maxpooling
#(
    .WIDTH(WIDTH0),
    .HEIGHT(HEIGHT0),
    .W_WIDTH(W_WIDTH0),
    .W_HEIGHT(W_HEIGHT0),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS)
)
pool_0to1
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(1),
    .in_pixels(ext_out),
    .in_vcnt(ext_vcnt),
    .in_hcnt(ext_hcnt),
    .out_enable(pool_enable1),
    .out_pixels(pool_out1),
    .out_vcnt(pool_vcnt1),
    .out_hcnt(pool_hcnt1)
);
// LEVEL1 Left
wire                      rdc_enable1;
wire [V_BITW1-1:0]        rdc_vcnt1;
wire [H_BITW1-1:0]        rdc_hcnt1;
wire [0:OUT_BITW*UNITS-1] rdc_out1;
rdcnet
#(
    .HEIGHT(HEIGHT1),
    .WIDTH(WIDTH1),
    .W_HEIGHT(W_HEIGHT1),
    .W_WIDTH(W_WIDTH1),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
rdcnet_inst1
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(pool_enable1),
    .in_y(pool_out1),
    .in_vcnt(pool_vcnt1),
    .in_hcnt(pool_hcnt1),
    .out_enable(rdc_enable1),
    .out_y(rdc_out1),
    .out_vcnt(rdc_vcnt1),
    .out_hcnt(rdc_hcnt1)
);
wire                      pool_enable2;
wire [V_BITW2-1:0]        pool_vcnt2;
wire [H_BITW2-1:0]        pool_hcnt2;
wire [0:OUT_BITW*UNITS-1] pool_out2;
maxpooling
#(
    .WIDTH(WIDTH1),
    .HEIGHT(HEIGHT1),
    .W_WIDTH(W_WIDTH1),
    .W_HEIGHT(W_HEIGHT1),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS)
)
pool_1to2
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(rdc_enable1),
    .in_pixels(rdc_out1),
    .in_vcnt(rdc_vcnt1),
    .in_hcnt(rdc_hcnt1),
    .out_enable(pool_enable2),
    .out_pixels(pool_out2),
    .out_vcnt(pool_vcnt2),
    .out_hcnt(pool_hcnt2)
);

// LEVEL2 Left
wire                      rdc_enable2;
wire [V_BITW2-1:0]        rdc_vcnt2;
wire [H_BITW2-1:0]        rdc_hcnt2;
wire [0:OUT_BITW*UNITS-1] rdc_out2;
rdcnet
#(
    .HEIGHT(HEIGHT2),
    .WIDTH(WIDTH2),
    .W_HEIGHT(W_HEIGHT2),
    .W_WIDTH(W_WIDTH2),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
rdcnet_inst2
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(pool_enable2),
    .in_y(pool_out2),
    .in_vcnt(pool_vcnt2),
    .in_hcnt(pool_hcnt2),
    .out_enable(rdc_enable2),
    .out_y(rdc_out2),
    .out_vcnt(rdc_vcnt2),
    .out_hcnt(rdc_hcnt2)
);
wire                      pool_enable3;
wire [V_BITW3-1:0]        pool_vcnt3;
wire [H_BITW3-1:0]        pool_hcnt3;
wire [0:OUT_BITW*UNITS-1] pool_out3;
maxpooling
#(
    .WIDTH(WIDTH2),
    .HEIGHT(HEIGHT2),
    .W_WIDTH(W_WIDTH2),
    .W_HEIGHT(W_HEIGHT2),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS)
)
pool_2to3
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(rdc_enable2),
    .in_pixels(rdc_out2),
    .in_vcnt(rdc_vcnt2),
    .in_hcnt(rdc_hcnt2),
    .out_enable(pool_enable3),
    .out_pixels(pool_out3),
    .out_vcnt(pool_vcnt3),
    .out_hcnt(pool_hcnt3)
);

// LEVEL3 Left
wire                      rdc_enable3;
wire [V_BITW3-1:0]        rdc_vcnt3;
wire [H_BITW3-1:0]        rdc_hcnt3;
wire [0:OUT_BITW*UNITS-1] rdc_out3;
rdcnet
#(
    .HEIGHT(HEIGHT3),
    .WIDTH(WIDTH3),
    .W_HEIGHT(W_HEIGHT3),
    .W_WIDTH(W_WIDTH3),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
rdcnet_inst3
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(pool_enable3),
    .in_y(pool_out3),
    .in_vcnt(pool_vcnt3),
    .in_hcnt(pool_hcnt3),
    .out_enable(rdc_enable3),
    .out_y(rdc_out3),
    .out_vcnt(rdc_vcnt3),
    .out_hcnt(rdc_hcnt3)
);
wire                      pool_enable4;
wire [V_BITW4-1:0]        pool_vcnt4;
wire [H_BITW4-1:0]        pool_hcnt4;
wire [0:OUT_BITW*UNITS-1] pool_out4;
maxpooling
#(
    .WIDTH(WIDTH3),
    .HEIGHT(HEIGHT3),
    .W_WIDTH(W_WIDTH3),
    .W_HEIGHT(W_HEIGHT3),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS)
)
pool_3to4
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(rdc_enable3),
    .in_pixels(rdc_out3),
    .in_vcnt(rdc_vcnt3),
    .in_hcnt(rdc_hcnt3),
    .out_enable(pool_enable4),
    .out_pixels(pool_out4),
    .out_vcnt(pool_vcnt4),
    .out_hcnt(pool_hcnt4)
);

// LEVEL4
wire                      rdc_enable4;
wire [V_BITW4-1:0]        rdc_vcnt4;
wire [H_BITW4-1:0]        rdc_hcnt4;
wire [0:OUT_BITW*UNITS-1] rdc_out4;
rdcnet
#(
    .HEIGHT(HEIGHT4),
    .WIDTH(WIDTH4),
    .W_HEIGHT(W_HEIGHT4),
    .W_WIDTH(W_WIDTH4),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
rdcnet_inst4
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(pool_enable4),
    .in_y(pool_out4),
    .in_vcnt(pool_vcnt4),
    .in_hcnt(pool_hcnt4),
    .out_enable(rdc_enable4),
    .out_y(rdc_out4),
    .out_vcnt(rdc_vcnt4),
    .out_hcnt(rdc_hcnt4)
);


// LEVEL3 Right
wire                      unpool_enable3;
wire [V_BITW3-1:0]        unpool_vcnt3;
wire [H_BITW3-1:0]        unpool_hcnt3;
wire [0:OUT_BITW*UNITS-1] unpool_out3;
unpooling
#(
    .WIDTH(WIDTH4),
    .HEIGHT(HEIGHT4),
    .W_WIDTH(W_WIDTH4),
    .W_HEIGHT(W_HEIGHT4),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS),
    .LEVEL(3)
)
unpool_4to3
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(rdc_enable4),
    .in_pixels(rdc_out4),
    .in_vcnt(rdc_vcnt4),
    .in_hcnt(rdc_hcnt4),
    .out_enable(unpool_enable3),
    .out_pixels(unpool_out3),
    .out_vcnt(unpool_vcnt3),
    .out_hcnt(unpool_hcnt3)
);
wire [0:OUT_BITW*UNITS-1] buf_out3;
ram
#(
    .WORD_SIZE(OUT_BITW*UNITS),
    .RAM_SIZE(W_HEIGHT3*W_WIDTH3)
)
buf3
(
    .wr_clock(clock),
    .rd_clock(clock),
    .wr_en(rdc_enable3),
    .wr_addr({rdc_vcnt3,rdc_hcnt3}),
    .wr_data(rdc_out3),
    .rd_addr({unpool_vcnt3,unpool_hcnt3}),
    .rd_data(buf_out3)
);

reg                      unpool_enable3_reg;
reg [V_BITW3-1:0]        unpool_vcnt3_reg;
reg [H_BITW3-1:0]        unpool_hcnt3_reg;
reg [0:OUT_BITW*UNITS-1] unpool_out3_reg;

always @(posedge clock)begin
    unpool_enable3_reg <= unpool_enable3;
    unpool_vcnt3_reg   <= unpool_vcnt3;
    unpool_hcnt3_reg   <= unpool_hcnt3;
    unpool_out3_reg    <= unpool_out3;
end

wire                      itg_enable3;
wire [V_BITW3-1:0]        itg_vcnt3;
wire [H_BITW3-1:0]        itg_hcnt3;
wire [0:OUT_BITW*UNITS-1] itg_out3;
itgnet
#(
    .HEIGHT(HEIGHT3),
    .WIDTH(WIDTH3),
    .W_HEIGHT(W_HEIGHT3),
    .W_WIDTH(W_WIDTH3),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
itgnet_inst3
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(unpool_enable3_reg),
    .in_y({unpool_out3_reg,buf_out3}),
    .in_vcnt(unpool_vcnt3_reg),
    .in_hcnt(unpool_hcnt3_reg),
    .out_enable(itg_enable3),
    .out_y(itg_out3),
    .out_vcnt(itg_vcnt3),
    .out_hcnt(itg_hcnt3)
);

// LEVEL2 Right
wire                      unpool_enable2;
wire [V_BITW2-1:0]        unpool_vcnt2;
wire [H_BITW2-1:0]        unpool_hcnt2;
wire [0:OUT_BITW*UNITS-1] unpool_out2;
unpooling
#(
    .WIDTH(WIDTH3),
    .HEIGHT(HEIGHT3),
    .W_WIDTH(W_WIDTH3),
    .W_HEIGHT(W_HEIGHT3),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS),
    .LEVEL(2)
)
unpool_3to2
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(itg_enable3),
    .in_pixels(itg_out3),
    .in_vcnt(itg_vcnt3),
    .in_hcnt(itg_hcnt3),
    .out_enable(unpool_enable2),
    .out_pixels(unpool_out2),
    .out_vcnt(unpool_vcnt2),
    .out_hcnt(unpool_hcnt2)
);
wire [0:OUT_BITW*UNITS-1] buf_out2;
ram
#(
    .WORD_SIZE(OUT_BITW*UNITS),
    .RAM_SIZE(W_HEIGHT2*W_WIDTH2)
)
buf2
(
    .wr_clock(clock),
    .rd_clock(clock),
    .wr_en(rdc_enable2),
    .wr_addr({rdc_vcnt2,rdc_hcnt2}),
    .wr_data(rdc_out2),
    .rd_addr({unpool_vcnt2,unpool_hcnt2}),
    .rd_data(buf_out2)
);
reg                      unpool_enable2_reg;
reg [V_BITW2-1:0]        unpool_vcnt2_reg;
reg [H_BITW2-1:0]        unpool_hcnt2_reg;
reg [0:OUT_BITW*UNITS-1] unpool_out2_reg;

always @(posedge clock)begin
    unpool_enable2_reg <= unpool_enable2;
    unpool_vcnt2_reg   <= unpool_vcnt2;
    unpool_hcnt2_reg   <= unpool_hcnt2;
    unpool_out2_reg    <= unpool_out2;
end

wire                      itg_enable2;
wire [V_BITW2-1:0]        itg_vcnt2;
wire [H_BITW2-1:0]        itg_hcnt2;
wire [0:OUT_BITW*UNITS-1] itg_out2;
itgnet
#(
    .HEIGHT(HEIGHT2),
    .WIDTH(WIDTH2),
    .W_HEIGHT(W_HEIGHT2),
    .W_WIDTH(W_WIDTH2),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
itgnet_inst2
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(unpool_enable2_reg),
    .in_y({unpool_out2_reg,buf_out2}),
    .in_vcnt(unpool_vcnt2_reg),
    .in_hcnt(unpool_hcnt2_reg),
    .out_enable(itg_enable2),
    .out_y(itg_out2),
    .out_vcnt(itg_vcnt2),
    .out_hcnt(itg_hcnt2)
);

// LEVEL1 Right
wire                      unpool_enable1;
wire [V_BITW1-1:0]        unpool_vcnt1;
wire [H_BITW1-1:0]        unpool_hcnt1;
wire [0:OUT_BITW*UNITS-1] unpool_out1;
unpooling
#(
    .WIDTH(WIDTH2),
    .HEIGHT(HEIGHT2),
    .W_WIDTH(W_WIDTH2),
    .W_HEIGHT(W_HEIGHT2),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS),
    .LEVEL(1)
)
unpool_2to1
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(itg_enable2),
    .in_pixels(itg_out2),
    .in_vcnt(itg_vcnt2),
    .in_hcnt(itg_hcnt2),
    .out_enable(unpool_enable1),
    .out_pixels(unpool_out1),
    .out_vcnt(unpool_vcnt1),
    .out_hcnt(unpool_hcnt1)
);
wire [0:OUT_BITW*UNITS-1] buf_out1;
ram
#(
    .WORD_SIZE(OUT_BITW*UNITS),
    .RAM_SIZE(W_HEIGHT1*W_WIDTH1)
)
buf1
(
    .wr_clock(clock),
    .rd_clock(clock),
    .wr_en(rdc_enable1),
    .wr_addr({rdc_vcnt1,rdc_hcnt1}),
    .wr_data(rdc_out1),
    .rd_addr({unpool_vcnt1,unpool_hcnt1}),
    .rd_data(buf_out1)
);
reg                      unpool_enable1_reg;
reg [V_BITW1-1:0]        unpool_vcnt1_reg;
reg [H_BITW1-1:0]        unpool_hcnt1_reg;
reg [0:OUT_BITW*UNITS-1] unpool_out1_reg;

always @(posedge clock)begin
    unpool_enable1_reg <= unpool_enable1;
    unpool_vcnt1_reg   <= unpool_vcnt1;
    unpool_hcnt1_reg   <= unpool_hcnt1;
    unpool_out1_reg    <= unpool_out1;
end

wire                      itg_enable1;
wire [V_BITW1-1:0]        itg_vcnt1;
wire [H_BITW1-1:0]        itg_hcnt1;
wire [0:OUT_BITW*UNITS-1] itg_out1;
itgnet
#(
    .HEIGHT(HEIGHT1),
    .WIDTH(WIDTH1),
    .W_HEIGHT(W_HEIGHT1),
    .W_WIDTH(W_WIDTH1),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
itgnet_inst1
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(unpool_enable1_reg),
    .in_y({unpool_out1_reg,buf_out1}),
    .in_vcnt(unpool_vcnt1_reg),
    .in_hcnt(unpool_hcnt1_reg),
    .out_enable(itg_enable1),
    .out_y(itg_out1),
    .out_vcnt(itg_vcnt1),
    .out_hcnt(itg_hcnt1)
);

// LEVEL0 Right
wire                      unpool_enable0;
wire [V_BITW0-1:0]        unpool_vcnt0;
wire [H_BITW0-1:0]        unpool_hcnt0;
wire [0:OUT_BITW*UNITS-1] unpool_out0;
unpooling
#(
    .WIDTH(WIDTH1),
    .HEIGHT(HEIGHT1),
    .W_WIDTH(W_WIDTH1),
    .W_HEIGHT(W_HEIGHT1),
    .FIXED_BITW(OUT_BITW),
    .UNITS(UNITS),
    .LEVEL(0)
)
unpool_1to0
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(itg_enable1),
    .in_pixels(itg_out1),
    .in_vcnt(itg_vcnt1),
    .in_hcnt(itg_hcnt1),
    .out_enable(unpool_enable0),
    .out_pixels(unpool_out0),
    .out_vcnt(unpool_vcnt0),
    .out_hcnt(unpool_hcnt0)
);
wire [0:OUT_BITW*UNITS-1] buf_out0;
ram
#(
    .WORD_SIZE(OUT_BITW*UNITS),
    .RAM_SIZE(W_HEIGHT0*W_WIDTH0)
)
buf0
(
    .wr_clock(clock),
    .rd_clock(clock),
    .wr_en(1),
    .wr_addr({ext_vcnt,ext_hcnt}),
    .wr_data(ext_out),
    .rd_addr({unpool_vcnt0,unpool_hcnt0}),
    .rd_data(buf_out0)
);
reg                      unpool_enable0_reg;
reg [V_BITW0-1:0]        unpool_vcnt0_reg;
reg [H_BITW0-1:0]        unpool_hcnt0_reg;
reg [0:OUT_BITW*UNITS-1] unpool_out0_reg;

always @(posedge clock)begin
    unpool_enable0_reg <= unpool_enable0;
    unpool_vcnt0_reg   <= unpool_vcnt0;
    unpool_hcnt0_reg   <= unpool_hcnt0;
    unpool_out0_reg    <= unpool_out0;
end

wire                      itg_enable0;
wire [V_BITW0-1:0]        itg_vcnt0;
wire [H_BITW0-1:0]        itg_hcnt0;
wire [0:OUT_BITW*UNITS-1] itg_out0;
itgnet
#(
    .HEIGHT(HEIGHT0),
    .WIDTH(WIDTH0),
    .W_HEIGHT(W_HEIGHT0),
    .W_WIDTH(W_WIDTH0),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(3)
)
itgnet_inst0
(
    .clock(clock),
    .n_rst(n_rst),
    .in_enable(unpool_enable0_reg),
    .in_y({unpool_out0_reg,buf_out0}),
    .in_vcnt(unpool_vcnt0_reg),
    .in_hcnt(unpool_hcnt0_reg),
    .out_enable(itg_enable0),
    .out_y(itg_out0),
    .out_vcnt(itg_vcnt0),
    .out_hcnt(itg_hcnt0)
);

wire [0:OUT_BITW*4-1] end_y;
wire [V_BITW0-1:0]    end_vcnt;
wire [H_BITW0-1:0]    end_hcnt;
conv_one
#(
    .HEIGHT(HEIGHT0),
    .WIDTH(WIDTH0),
    .W_HEIGHT(W_HEIGHT0),
    .W_WIDTH(W_WIDTH0),
    .UINT_BITW(OUT_BITW),
    .PATCH_SIZE(1)
)
(
    .clock(clock),
    .n_rst(n_rst),
    .in_y(itg_out0),
    .in_vcnt(itg_vcnt0),
    .in_hcnt(itg_hcnt0),
    .out_y(end_y),
    .out_hcnt(end_hcnt),
    .out_vcnt(end_vcnt)
);

reg signed [OUT_BITW-1:0]  a,b,c,d;
reg signed [OUT_BITW-1:0]  more0,more1;
reg        [1:0]             win1,win2,win3;
reg [0:OUT_BITW-1]         max;
reg [V_BITW0-1:0]    end_vcnt1, end_vcnt2, end_vcnt3;
reg [H_BITW0-1:0]    end_hcnt1, end_hcnt2, end_hcnt3; 

assign out_y    = win3;
assign out_vcnt = end_vcnt3;
assign out_hcnt = end_hcnt3;

always @(posedge clock) begin
    a <= end_y[0          +: OUT_BITW];
    b <= end_y[OUT_BITW   +: OUT_BITW];
    c <= end_y[OUT_BITW*2 +: OUT_BITW];
    d <= end_y[OUT_BITW*3 +: OUT_BITW];

    end_vcnt1 <= end_vcnt;
    end_hcnt1 <= end_hcnt;
    end_vcnt2 <= end_vcnt1;
    end_hcnt2 <= end_hcnt1;
    end_vcnt3 <= end_vcnt2;
    end_hcnt3 <= end_hcnt2;

    if(a > b)begin
        more0 <= a;
        win1  <= 2'b00;
    end else begin
        more0 <= b;
        win1  <= 2'b01;
    end
    if(c > d)begin
        more1 <= c;
        win1  <= 2'b10;
    end else begin
        more1 <= d;
        win1  <= 2'b11;
    end
    if(more0 > more1)begin
        win3 <= win1;
    end else begin
        win3 <= win2;
    end
end

// functions ---------------------------------------------------------------
// calculates ceil(log2(value))
function integer log2;
    input [63:0] value;
    begin
        value = value - 1;
        for ( log2 = 0; value > 0; log2 = log2 + 1 )
            value = value >> 1;
    end
endfunction

endmodule
`default_nettype wire
