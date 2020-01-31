//-----------------------------------------------------------------------------
// <cnn-fixed>
//  - Convolutional Neural Network (CNN) module
//-----------------------------------------------------------------------------
// Version 4.10 (June 27, 2018)
//  - Compatible with color images (YCbCr)
//  - Code refinement
//-----------------------------------------------------------------------------
// (C) 2018 Taito Manabe, all rights reserved.
//-----------------------------------------------------------------------------
// kfujita
// <RdcNet-fixed>
//  -
//-----------------------------------------------------------------------------
// Version 1.0 (12_18, 2019)
//-----------------------------------------------------------------------------
`default_nettype none
`timescale 1ns/1ns

module conv_one
#(  parameter integer HEIGHT     = -1,
    parameter integer WIDTH      = -1,
    parameter integer W_HEIGHT   = -1,
    parameter integer W_WIDTH    = -1,
    parameter integer UINT_BITW  = -1, // input image bit width ( Fixed 8 ... ?)
    parameter integer PATCH_SIZE = -1 )
(   clock, n_rst,
    in_y,  in_vcnt,  in_hcnt,
    out_y,  out_vcnt, out_hcnt );

// parameters --------------------------------------------------------------
// filter size
localparam integer L1_FLT_SIZE = 1;
//localparam integer L2_FLT_SIZE = 3;
//localparam integer L3_FLT_SIZE = 3;
// number of units
localparam integer L1_UNITS    = 12; // the same prev Net output
//localparam integer L2_UNITS    = 12;
//localparam integer L3_UNITS    = 12;
// fixed-point format
localparam integer L1_INT_BITW = 5;  // the same prev Net output
//localparam integer L2_INT_BITW = 5;
//localparam integer L3_INT_BITW = 5;
localparam integer FRAC_BITW   = 8;  // must be = 8 (???)

// other parameters --------------------------------------------------------
localparam integer L1_FIXED_BITW = L1_INT_BITW + FRAC_BITW;
//localparam integer L2_FIXED_BITW = L2_INT_BITW + FRAC_BITW;
//localparam integer L3_FIXED_BITW = L3_INT_BITW + FRAC_BITW;
localparam integer V_BITW        = log2(W_HEIGHT);
localparam integer H_BITW        = log2(W_WIDTH);
localparam integer ADJUST        = (PATCH_SIZE - 1) * 2;

// inputs/outputs ----------------------------------------------------------
input wire                               clock, n_rst;
//input wire [UINT_BITW-1:0]             in_y;
input wire [0:L1_FIXED_BITW*L1_UNITS-1]  in_y;
input wire [V_BITW-1:0]                  in_vcnt;
input wire [H_BITW-1:0]                  in_hcnt;
//output reg [UINT_BITW-1:0]             out_y;
//output wire [0:L3_FIXED_BITW*L3_UNITS-1] out_y;
output wire [0:L1_FIXED_BITW*4-1]        out_y;
output wire [V_BITW-1:0]                 out_vcnt;
output wire [H_BITW-1:0]                 out_hcnt;

// imports filters/biases --------------------------------------------------
`include "/home/users/kfujita/resol_hdl/segment/src/param/unit12_lkh_fixed_param.txt"

// [layer 1] ---------------------------------------------------------------
wire [0:L1_FIXED_BITW*L1_UNITS-1]   in_fixed;
wire [0:L1_FIXED_BITW*4-1]   l1_out;
wire [V_BITW-1:0]                   l1_vcnt;
wire [H_BITW-1:0]                   l1_hcnt;

generate
if(FRAC_BITW == UINT_BITW)
    assign in_fixed = in_y;
else if(UINT_BITW < FRAC_BITW) begin
    localparam [FRAC_BITW-UINT_BITW-1:0] EXPAND = 0;
    assign in_fixed = {in_y, EXPAND};
end
endgenerate


layer
#(  .WIDTH(WIDTH),          .HEIGHT(HEIGHT),
    .W_WIDTH(W_WIDTH),      .W_HEIGHT(W_HEIGHT),
    .UINT_BITW(UINT_BITW),  .FRAC_BITW(FRAC_BITW),
    .PREV_UNITS(L1_UNITS),  .PREV_INT_BITW(L1_INT_BITW),
    .UNITS(4),       .INT_BITW(L1_INT_BITW),
    .FLT_SIZE(L1_FLT_SIZE), .FLT(L1_FLT),           .BIAS(L1_BIAS),
    .LEVEL(0)  )
layer_1
(   .clock(clock),        .n_rst(n_rst),
    .in_enable(1),
    .in_pixels(in_fixed),
    .in_vcnt(in_vcnt),    .in_hcnt(in_hcnt),
    .out_enable(),
    .out_pixels(l1_out),
    .out_vcnt(l1_vcnt),   .out_hcnt(l1_hcnt)                 );

assign out_y    = l1_out;
assign out_vcnt = l1_vcnt;
assign out_hcnt = l1_hcnt;
// [layer 4] ---------------------------------------------------------------

// converts from fixed-point to uint8 --------------------------------------
/*
wire [UINT_BITW:0]      result;
assign result 
= l4_out[L4_FIXED_BITW-1] == 1 ? 0 :
FRAC_BITW == UINT_BITW ? l4_out[L4_FIXED_BITW-2:0] :
((l4_out[L4_FIXED_BITW-2:0] >> (FRAC_BITW - UINT_BITW - 1)) + 1) >> 1;
always @(posedge clock)
    out_y <= (result[UINT_BITW] == 1'b1) ? (-1) : result;
*/
// coordinates adjustment --------------------------------------------------
/*
coord_adjuster
#( .HEIGHT(W_HEIGHT), .WIDTH(W_WIDTH), 
   .LATENCY((PATCH_SIZE * 2 - 2) * (W_WIDTH + 1) + 1) )
ca_0
(  .clock(clock), 
   .in_vcnt(l4_vcnt),   .in_hcnt(l4_hcnt),
   .out_vcnt(out_vcnt), .out_hcnt(out_hcnt) );   
*/
// delay for cb/cr channels ------------------------------------------------
/*
delay
#( .BIT_WIDTH(UINT_BITW * 2), 
   .LATENCY((PATCH_SIZE * 2 - 2) * (W_WIDTH + 1) + 1) )
dl_0
(  .clock(clock),            .n_rst(n_rst), 
   .in_data({l4_cb, l4_cr}), .out_data({out_cb, out_cr}) );
*/
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
