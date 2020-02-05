//-----------------------------------------------------------------------------
// <layer-fixed>
//  - Composes one layer of CNN
//  - Generates network input images with the stream architecture,
//    applies flips corresponding to the coordinates, obtains results
//    using <forward> module and converts them to binary format
//-----------------------------------------------------------------------------
// Version 4.10 (June 27, 2018)
//  - Compatible with color images (YCbCr)
//  - Code refinement
//-----------------------------------------------------------------------------
// (C) 2018 Taito Manabe, all rights reserved.
//-----------------------------------------------------------------------------
// maxpooling kfujita
//

`default_nettype none
`timescale 1ns/1ns
`define UL 3'b000
`define UR 3'b001
`define LL 3'b010
`define LR 3'b011
`define EN_WAIT 3'b100
`define LN_WAIT 3'b101

module unpooling
#(  parameter integer WIDTH         = -1,
    parameter integer HEIGHT        = -1,
    parameter integer W_WIDTH       = -1,
    parameter integer W_HEIGHT      = -1,
    parameter integer FIXED_BITW    = -1,
    parameter integer UNITS         = -1,
    parameter integer LEVEL         = -1
)
(   clock,      n_rst,
    in_enable,
    in_pixels, in_vcnt,  in_hcnt,
    out_enable,
    out_pixels, out_vcnt, out_hcnt );

// local parameters --------------------------------------------------------
localparam integer FLT_SIZE   = 2;                   // 2x2 max pooling
//localparam integer IN_PIXS    = FLT_SIZE * FLT_SIZE * UNITS;
//localparam integer LATENCY    = 3 + log2(IN_PIXS);  // meccazyuuouyade tyanntosina
//localparam integer LATENCY    = 4;  /////////////////// mecca zisin naiyo!!!!
//localparam integer PATCH_SIZE = FLT_SIZE;
//localparam integer PATCH_BITW = PATCH_SIZE * PATCH_SIZE * FIXED_BITW;
localparam integer V_BITW     = log2(W_HEIGHT);
localparam integer H_BITW     = log2(W_WIDTH);
// LEVEL BUF parameter
// 640 は入力時のWIDTHです。こうしないと正しく遅延できない？
localparam integer FIRST_WIDTH = W_WIDTH;
localparam integer UppR_BUF    =  1 << LEVEL;
localparam integer LowL_BUF    = (1 << LEVEL) * FIRST_WIDTH;
localparam integer LowR_BUF    = UppR_BUF + LowL_BUF;
localparam integer BUF_BITW    = log2(UppR_BUF);
localparam integer FIRST_WBITW = log2(FIRST_WIDTH);
localparam integer FIRST_TIME  = FIRST_WIDTH*UppR_BUF;
localparam integer FIRST_TIME_WBITW = log2(FIRST_TIME);
// inputs/outputs ----------------------------------------------------------
input wire                             clock, n_rst, in_enable;
input wire [0:FIXED_BITW*UNITS-1]      in_pixels;
input wire [V_BITW-1:0]                in_vcnt;
input wire [H_BITW-1:0]                in_hcnt;
output wire                            out_enable;
output wire [0:FIXED_BITW*UNITS-1]     out_pixels;
output wire [V_BITW-1:0]               out_vcnt;    //[log2(V_BITW/2)-1:0] ?
output wire [H_BITW-1:0]               out_hcnt;    //[log2(H_BITW/2)-1:0] ?
// -------------------------------------------------------------------------
// genvar      p, v, h, m;

// delay pixels
wire [0:FIXED_BITW*UNITS-1]     uppl_out_pixels;
wire [0:FIXED_BITW*UNITS-1]     uppr_out_pixels;
wire [0:FIXED_BITW*UNITS-1]     lowl_out_pixels;
wire [0:FIXED_BITW*UNITS-1]     lowr_out_pixels;
reg  [0:FIXED_BITW*UNITS-1]     in_pixels_reg;
reg                             in_enable_reg;
reg                             state_enable_reg;
reg                             line_enable_reg;

reg [2:0] state_reg;

reg [H_BITW-1:0]                hcnt_reg;
reg [V_BITW-1:0]                vcnt_reg;
wire [V_BITW-1:0]               uppl_out_vcnt;
wire [H_BITW-1:0]               uppl_out_hcnt;
wire [V_BITW-1:0]               uppr_out_vcnt;
wire [H_BITW-1:0]               uppr_out_hcnt;
wire [V_BITW-1:0]               lowl_out_vcnt;
wire [H_BITW-1:0]               lowl_out_hcnt;
wire [V_BITW-1:0]               lowr_out_vcnt;
wire [H_BITW-1:0]               lowr_out_hcnt;

reg [V_BITW-1:0]               out_vcnt_reg;
reg [H_BITW-1:0]               out_hcnt_reg;

assign out_enable = LEVEL==0 ? 1'b1
                  : LEVEL==1 ? &out_hcnt[0] && &out_vcnt[0]
                  : LEVEL==2 ? &out_hcnt[1:0] && &out_vcnt[1:0]
                  : LEVEL==3 ? &out_hcnt[2:0] && &out_vcnt[2:0]
                  :            &out_hcnt[3:0] && &out_vcnt[3:0];

assign out_pixels = LEVEL==0 ? out_hcnt[0]==1'b0 && out_vcnt[0]==1'b0 ? in_pixels
                             : out_hcnt[0]==1'b1 && out_vcnt[0]==1'b0 ? uppr_out_pixels
                             : out_hcnt[0]==1'b0 && out_vcnt[0]==1'b1 ? lowl_out_pixels
                             :                                          lowr_out_pixels
                  : LEVEL==1 ? &out_hcnt[0] && &out_vcnt[0] ? (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b0) ? in_pixels
                                                            : (out_hcnt[LEVEL]==1'b1 && out_vcnt[LEVEL]==1'b0) ? uppr_out_pixels
                                                            : (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b1) ? lowl_out_pixels
                                                            :                                                    lowl_out_pixels
                             : 'b1
                  : LEVEL==2 ? &out_hcnt[1:0] && &out_vcnt[1:0] ? (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b0) ? in_pixels
                                                                : (out_hcnt[LEVEL]==1'b1 && out_vcnt[LEVEL]==1'b0) ? uppr_out_pixels
                                                                : (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b1) ? lowl_out_pixels
                                                                :                                                    lowl_out_pixels
                             : 'b1
                  : LEVEL==3 ? &out_hcnt[2:0] && &out_vcnt[2:0] ? (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b0) ? in_pixels
                                                                : (out_hcnt[LEVEL]==1'b1 && out_vcnt[LEVEL]==1'b0) ? uppr_out_pixels
                                                                : (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b1) ? lowl_out_pixels
                                                                :                                                    lowl_out_pixels
                             : 'b1
                  :            &out_hcnt[3:0] && &out_vcnt[3:0] ? (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b0) ? in_pixels
                                                                : (out_hcnt[LEVEL]==1'b1 && out_vcnt[LEVEL]==1'b0) ? uppr_out_pixels
                                                                : (out_hcnt[LEVEL]==1'b0 && out_vcnt[LEVEL]==1'b1) ? lowl_out_pixels
                                                                :                                                    lowl_out_pixels
                             : 'b1;

assign out_hcnt = lowr_out_hcnt;
assign out_vcnt = lowr_out_vcnt;

delay
#(  .BIT_WIDTH(FIXED_BITW*UNITS),
    .LATENCY(UppR_BUF)
)
delay_UppR
(   .clock(clock),  .n_rst(n_rst),
    .enable(1'b1),
    .in_data(in_pixels), .out_data(uppr_out_pixels)
);
delay
#(  .BIT_WIDTH(FIXED_BITW*UNITS),
    .LATENCY(LowL_BUF)
)
delay_LowL
(   .clock(clock),  .n_rst(n_rst),
    .enable(1'b1),
    .in_data(in_pixels), .out_data(lowl_out_pixels)
);
delay
#(  .BIT_WIDTH(FIXED_BITW*UNITS),
    .LATENCY(LowR_BUF)
)
delay_LowR
(   .clock(clock),  .n_rst(n_rst),
    .enable(1'b1),
    .in_data(in_pixels), .out_data(lowr_out_pixels)
);
coord_adjuster
#(
   .HEIGHT(W_HEIGHT), .WIDTH(W_WIDTH), .LATENCY(LowR_BUF)
)
ca_1
(
    .clock(clock), .in_vcnt(in_vcnt), .in_hcnt(in_hcnt),
    .out_vcnt(lowr_out_vcnt), .out_hcnt(lowr_out_hcnt)
);

// common functions --------------------------------------------------------
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
